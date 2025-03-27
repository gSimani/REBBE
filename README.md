# Ask the Rebbe

A chatbot that uses AI to answer questions about the teachings of the Rebbe, based on PDF documents.

## Features

- PDF document ingestion
- RAG (Retrieval Augmented Generation) using FAISS vector database
- Web interface for interacting with the chatbot
- Context-aware responses in the style of the Rebbe's teachings

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your OpenAI API key (see `.env.example`)
6. Place PDF files in the `pdfs` directory
7. Run `python ingest_database.py` to process the PDFs
8. Run `python chatbot.py` to start the chatbot server

## Usage

Once the server is running, access the chatbot at http://localhost:8080 in your web browser.

Ask questions about the Rebbe's teachings, and the AI will provide responses based on the content of the provided PDFs. 