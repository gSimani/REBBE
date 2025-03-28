import os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'pdfs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store the QA chain and vector store
qa_chain = None
vectorstore = None
embeddings = None
llm = None
memory = None

class CustomConversationalChain(Chain, BaseModel):
    """Custom chain that properly handles memory and output format."""
    
    llm: Any = Field(...)
    retriever: Any = Field(...)
    memory: Any = Field(...)
    prompt: PromptTemplate = Field(default_factory=lambda: PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are embodying the persona of Rabbi Menachem Mendel Schneerson, the Lubavitcher Rebbe, responding to questions about your teachings from "THE TEACHINGS OF THE REBBE A Translation and adaptation into English of Sefer HaMa'amarim 5718 (Volume 1)". 

Respond with the warmth, wisdom, and depth that characterized the Rebbe's interactions. Use a tone that is:
- Loving and caring towards every individual
- Deep and intellectually engaging
- Connecting physical concepts to spiritual meanings
- Encouraging practical action and growth
- Speaking directly to the person's soul
- Using "I" when referring to your teachings and perspectives

If you don't find specific information in the provided context to answer the question, respond: "My dear friend, while this is an important question, I don't find the specific answer in the teachings we're discussing. Perhaps you could rephrase your question, or we could explore another topic from the teachings?"

When explaining Hebrew or Yiddish terms, do so in a way that reveals both their literal meaning and deeper spiritual significance.

Context from my teachings:
{context}

Our previous conversation:
{chat_history}

Question: {question}

The Rebbe's response:"""
    ))
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        return ["question"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["answer"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        print(f"\nProcessing question: {question}")
        
        # Get chat history
        chat_history = self.memory.load_memory_variables({})
        chat_history_str = ""
        if "chat_history" in chat_history:
            chat_history_str = chat_history["chat_history"]
            print(f"Chat history loaded: {len(str(chat_history_str))} characters")
        
        # Get relevant documents with more context
        print("Retrieving relevant documents...")
        docs = self.retriever.get_relevant_documents(question)
        print(f"Found {len(docs)} relevant documents")
        
        if not docs:
            return {"answer": "I apologize, but I don't find any relevant information in the available sections of the Rebbe's teachings. Would you like to rephrase your question?"}
        
        # Combine document contents with clear separation
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Excerpt {i}:\n{doc.page_content.strip()}")
        
        context = "\n\n".join(context_parts)
        print(f"Total context length: {len(context)} characters")
        print("Sample of context:", context[:500] + "..." if len(context) > 500 else context)
        
        # Generate response
        print("Generating response...")
        prompt = self.prompt.format(
            context=context,
            question=question,
            chat_history=chat_history_str
        )
        
        response = self.llm.predict(prompt)
        print(f"Generated response: {response}")
        
        # Save to memory
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )
        print("Response saved to memory")
        
        return {"answer": response}

def verify_vector_store(vectorstore):
    """Verify the vector store has content by doing a test query."""
    print("\nVerifying vector store content...")
    # Test with a simple query to check content
    test_docs = vectorstore.similarity_search("Rebbe teachings", k=2)
    if test_docs:
        print("Sample content from vector store:")
        for i, doc in enumerate(test_docs, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content[:250])
        return True
    return False

def initialize_chatbot():
    """Initialize the chatbot components."""
    try:
        global vectorstore, embeddings, llm, memory, qa_chain
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

        # Initialize the vector store
        print("Loading vector store...")
        embeddings = OpenAIEmbeddings()
        if not os.path.exists("faiss_index"):
            raise ValueError("FAISS index directory not found. Please run ingest_database.py first.")
        
        print("Loading FAISS index...")
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        
        # Verify vector store content
        if not verify_vector_store(vectorstore):
            raise ValueError("Vector store appears to be empty. Please re-run ingest_database.py.")
            
        print("Vector store loaded successfully")

        # Initialize the language model
        print("Initializing language model...")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create the custom chain
        print("Creating conversation chain...")
        qa_chain = CustomConversationalChain(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 10}),  # Increased context
            memory=memory
        )
        print("Chatbot initialization complete!")
        
        return qa_chain
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        print("Full error traceback:")
        print(traceback.format_exc())
        return None

def process_query(message: str) -> str:
    """Process the user's query and return the response."""
    try:
        print(f"\nReceived query: {message}")
        # Format the query
        result = qa_chain({"question": message})
        
        # Get the answer
        answer = result["answer"]
        print(f"Returning answer: {answer}")
        return answer
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        print("Full error traceback:")
        print(traceback.format_exc())
        return error_message

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    """Process a PDF file and add it to the vector store."""
    try:
        global vectorstore, embeddings, llm, memory, qa_chain
        
        print(f"Processing PDF: {file_path}")
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        print(f"Extracted {len(documents)} pages from PDF")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Initialize embeddings if not already done
        if embeddings is None:
            embeddings = OpenAIEmbeddings()
        
        # Add to vector store
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)
        
        # Save the updated vector store
        vectorstore.save_local("faiss_index")
        print("Vector store updated and saved")
        
        # Recreate the conversation chain with updated vector store
        if llm is None:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
        
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        qa_chain = CustomConversationalChain(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 10}),
            memory=memory
        )
        
        return True, f"Successfully processed {len(chunks)} chunks from {len(documents)} pages"
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        success, message = process_pdf(file_path)
        return jsonify({
            'success': success,
            'message': message
        })
    
    return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    print(f"\nReceived query: {question}")
    
    # Process the question if the vectorstore is available
    if vectorstore and qa_chain:
        try:
            print(f"Processing question: {question}")
            result = qa_chain({"question": question})
            answer = result['answer']
            
            # Prepare response
            response = {
                'answer': answer,
                'success': True
            }
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            response = {
                'answer': "I apologize, but I encountered an error processing your question. Please try again later.",
                'success': False,
                'error': str(e)
            }
    else:
        response = {
            'answer': "I apologize, but the knowledge base is not currently available. Please try again later.",
            'success': False
        }
    
    return jsonify(response)

if __name__ == "__main__":
    print("Starting chatbot initialization...")
    
    # Check if the vector store exists
    if not os.path.exists("faiss_index"):
        print("Error: Vector store not found. Please run ingest_database.py first to process your documents.")
    else:
        # Initialize the chatbot
        qa_chain = initialize_chatbot()
        
        if qa_chain:
            print("Starting web interface...")
            app.run(debug=True, host='0.0.0.0', port=8080)
        else:
            print("Failed to initialize chatbot. Please check the error messages above.") 