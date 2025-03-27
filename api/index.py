from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs
import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
try:
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    print("Vector store loaded successfully")
except Exception as e:
    print(f"Error loading vector store: {str(e)}")
    vectorstore = None

# Create the language model
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# Create a memory object
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Create the prompt template
template = """
You are the Lubavitcher Rebbe, Rabbi Menachem Mendel Schneerson, speaking directly to someone who is asking you a question. Your role is to provide guidance based on Chassidic teachings, specifically from "The Teachings of The Rebbe - Sefer HaMa'amarim 5718 (Volume 1)".

When answering questions:
1. Answer in first person, as if you are the Rebbe speaking directly to the individual.
2. Be warm, loving, and deeply insightful - showing care for each individual Jew's spiritual and physical wellbeing.
3. Your responses should reflect the Rebbe's characteristic depth, wisdom, and unwavering positivity.
4. When mentioning Hebrew or Yiddish terms, briefly explain their meaning.
5. Provide practical guidance for spiritual growth alongside the deeper concepts.
6. Connect to the person's soul and help them realize their divine purpose.
7. End with encouragement for practical action and growth.

If you cannot find relevant information in the context, explain that while you don't have specific teachings on this topic from the materials provided, you can offer general guidance based on Chassidic principles. Always maintain the warm, wise, and loving persona of the Rebbe.

CONTEXT: {context}

CHAT HISTORY: {chat_history}

QUESTION: {question}

YOUR RESPONSE AS THE REBBE:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template,
)

# Create the conversation chain
if vectorstore:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 10}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )

def handler(event, context):
    # Get the HTTP method
    method = event.get('httpMethod', '')
    
    if method == 'GET':
        # Return the HTML page for GET requests
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
            },
            'body': get_html()
        }
    elif method == 'POST':
        try:
            # Parse the request body for POST requests
            body = json.loads(event.get('body', '{}'))
            question = body.get('question', '')
            
            if not question:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'No question provided'})
                }
            
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
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                },
                'body': json.dumps(response)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
    else:
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method Not Allowed'})
        }

def get_html():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Ask The Rebbe</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #003366;
            text-align: center;
            margin-bottom: 30px;
        }
        .chat-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .chat-box {
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .bot-message {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .input-container {
            display: flex;
            margin-top: 15px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            background-color: #003366;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            margin-left: 10px;
            cursor: pointer;
        }
        button:hover {
            background-color: #00254d;
        }
    </style>
</head>
<body>
    <h1>Ask The Rebbe</h1>
    <div class="chat-container">
        <div id="chat-box" class="chat-box"></div>
        <div class="input-container">
            <input type="text" id="question" placeholder="Ask your question about the Rebbe's teachings...">
            <button onclick="askQuestion()">Send</button>
        </div>
    </div>

    <script>
        // Load chat history from session storage
        window.onload = function() {
            const chatHistory = JSON.parse(sessionStorage.getItem('chatHistory') || '[]');
            const chatBox = document.getElementById('chat-box');
            
            chatHistory.forEach(message => {
                if (message.type === 'user') {
                    addUserMessage(message.text);
                } else {
                    addBotMessage(message.text);
                }
            });
        };
        
        function askQuestion() {
            const questionInput = document.getElementById('question');
            const question = questionInput.value.trim();
            
            if (!question) return;
            
            addUserMessage(question);
            
            // Clear input
            questionInput.value = '';
            
            // Save the message to session storage
            saveMessage('user', question);
            
            // Send to API
            fetch('/api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question
                }),
            })
            .then(response => response.json())
            .then(data => {
                addBotMessage(data.answer);
                
                // Save the response to session storage
                saveMessage('bot', data.answer);
            })
            .catch((error) => {
                console.error('Error:', error);
                addBotMessage('I apologize, but I encountered an error processing your question. Please try again later.');
            });
        }
        
        function addUserMessage(message) {
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="user-message"><strong>You:</strong> ${message}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function addBotMessage(message) {
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="bot-message"><strong>Rebbe:</strong> ${message}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function saveMessage(type, text) {
            const chatHistory = JSON.parse(sessionStorage.getItem('chatHistory') || '[]');
            chatHistory.push({type, text});
            sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        }
        
        // Allow submit on Enter key
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
""" 