# PDF Documents

Place your PDF files in this directory to make them available for ingestion by the chatbot.

## Processing

After adding PDF files to this directory, run the ingestion script:

```
python ingest_database.py
```

This will:
1. Extract text from all PDF files
2. Split the text into manageable chunks
3. Create embeddings for each chunk
4. Store the embeddings in a FAISS vector database

## Supported Formats

The system currently supports PDF files. Each PDF file should contain teachings or writings related to the Rebbe that you wish to make available through the chatbot. 