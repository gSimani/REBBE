import os
from typing import List
from dotenv import load_dotenv
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
print("Current working directory:", os.getcwd())
print("Loading environment variables...")
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key) if api_key else 0}")

def load_pdfs(pdfs_dir: str) -> List:
    """Load all PDF files from the pdfs directory."""
    documents = []
    for filename in os.listdir(pdfs_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdfs_dir, filename)
            print(f"\nLoading {filename}...")
            loader = PDFPlumberLoader(file_path)
            try:
                docs = loader.load()
                print(f"Extracted {len(docs)} pages from {filename}")
                total_chars = sum(len(doc.page_content) for doc in docs)
                print(f"Total characters extracted: {total_chars}")
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return documents

def split_documents(documents: List) -> List:
    """Split documents into chunks."""
    print("\nSplitting documents into chunks...")
    print("Chunk size: 300 characters")
    print("Chunk overlap: 100 characters")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    print(f"\nCreated {len(chunks)} chunks")
    print(f"Total characters in chunks: {total_chars}")
    print(f"Average chunk size: {total_chars / len(chunks):.2f} characters")
    
    # Print sample chunks
    print("\nSample chunks (first 2):")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk.page_content)} characters")
        print("Content preview:", chunk.page_content[:150], "...")
    
    return chunks

def create_vector_store(chunks: List, persist_directory: str):
    """Create and persist the vector store."""
    print("\nCreating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print(f"Saving vector store to {persist_directory}")
    vectorstore.save_local(persist_directory)
    return vectorstore

def main():
    print("\n=== PDF Processing and Vector Store Creation ===\n")
    
    # Create necessary directories if they don't exist
    os.makedirs('pdfs', exist_ok=True)

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    # Load and process PDFs
    print("Loading PDFs from pdfs directory...")
    documents = load_pdfs('pdfs')
    
    if not documents:
        print("No PDF files found in the 'pdfs' directory. Please add some PDF files and try again.")
        return

    print(f"\nFound {len(documents)} document(s) total.")
    chunks = split_documents(documents)
    
    print("\nCreating vector store...")
    vectorstore = create_vector_store(chunks, 'faiss_index')
    
    print(f"\nProcessing complete!")
    print(f"Created vector store with {len(chunks)} chunks")
    print("\nYou can now use chatbot.py to interact with your documents.")

if __name__ == "__main__":
    main() 