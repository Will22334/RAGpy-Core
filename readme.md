RAGpy is a lightweight, modular Retrieval-Augmented Generation (RAG) pipeline for Python. It provides a clear and testable architecture for document ingestion, chunking, embedding, retrieval, reranking, context compression, and grounded answer generation using Azure OpenAI and ChromaDB.

RAGpy is designed for developers who want a transparent, hackable RAG system without the complexity of large frameworks.

Features
• Modular ingestion pipeline for text, PDF, and DOCX documents
• Deterministic chunking and batching utilities for efficient embedding
• Azure OpenAI embeddings and chat completions
• ChromaDB vector database integration
• LLM-based reranking for improved retrieval quality
• Context compression to reduce token usage
• Fully monkeypatch-friendly design for offline testing
• Clean architecture suitable for extension and customization

Installation
From PyPI:
pip install ragpy

Quickstart Example

from ragpy.AzureOpenAIRelay import SetEmbeddingEndpointInfo, SetCompletionEndpointInfo
from ragpy.RAGOrchestrator import IngestFile, GenerateAnswer
from ragpy.VectorDatabase import OpenDatabase

#Set your Azure Model Information here:
api_Key = "<your-key-here>"

embed_Endpoint = "your-embedding-deployment-endpoint" # ex. https://aif-xxxxxx.services.ai.azure.com
embed_Deployment_Name = "your-embedding-deployment-name" #ex. "oai-xxx-embedding-dev" or "text-embedding-3-small"
embed_Api_Version = 'your-api-version' #ex. "2024-02-01"

chat_Endpoint = "your-response-deployment-endpoint" # ex. https://aif-xxxxxx.services.ai.azure.com
chat_Name = "your-response-deployment-name" #ex. "oai-xxxx-chat-dev" or "gpt-40-mini"
chat_Api_Version = "your-api-version" #ex. "2024-02-01 

#Set Azure Model Information
SetEmbeddingEndpointInfo(embed_Endpoint, embed_Deployment_Name, embed_Api_Version, api_Key)
SetCompletionEndpointInfo(chat_Endpoint, chat_Name, chat_Api_Version, api_Key)

#Open or create a persistent ChromaDB collection
OpenDatabase("testDB", "./vectorDB")

#Ingest a document
IngestFile("./ExampleFiles/Sample_TXT.txt", "testDB")

#Ask a question
answer = GenerateAnswer("What is RAG?")
print(answer)

How RAGpy Works

Ingestion
Load text, PDF, or DOCX using FileLoader
Chunk text using TextChunker
Batch chunks using ChunkBatcher
Generate embeddings with Azure OpenAI
Store vectors and metadata in ChromaDB

Retrieval
Embed the user query
Retrieve top-K candidates from the vector database
Reranking
Use an LLM-based reranker to reorder retrieved chunks by relevance

Compression
Summarize top chunks into a compact context block
Reduce token usage while preserving grounding

Answer Generation
Build a prompt using compressed context
Generate a grounded answer using Azure OpenAI

Project Structure
ragpy/
AzureOpenAIRelay.py
RAGOrchestrator.py
VectorDatabase.py
Reranker.py
ChunkCompressor.py
loaders/
FileLoader.py
TextChunker.py
batching/
ChunkBatcher.py

tests/
docs/

Requirements
• Python 3.9+
• chromadb
• numpy
• tiktoken
• pypdf
• python-docx
• openai (Azure OpenAI SDK)

Testing

RAGpy includes a full pytest suite.
All Azure calls are monkeypatch-friendly, allowing offline testing with mock LLMs.

Run tests:
pytest -q

Contributing

Contributions are welcome.
Please open an issue or submit a pull request on GitHub.

Planned enhancements include:
• Local embedding support (sentence-transformers)
• Hybrid retrieval (vector + keyword)
• Multimodal RAG (image + text)
• Evaluation tools for relevance and faithfulness
• Agentic RAG extensions

License
RAGpy is released under the MIT License.