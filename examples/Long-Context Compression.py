# -*- coding: utf-8 -*-
"""
Example 4: Demonstrate long-context compression in RAGpy.
"""

from ragpy.AzureOpenAIRelay import SetEmbeddingEndpointInfo, SetCompletionEndpointInfo
from ragpy.RAGOrchestrator import IngestFile, GenerateAnswer
from ragpy.VectorDatabase import OpenDatabase

# Azure Configuration
api_Key = "<your-key-here>"

embed_Endpoint = "your-embedding-deployment-endpoint" # ex. https://aif-xxxxxx.services.ai.azure.com
embed_Deployment_Name = "your-embedding-deployment-name" #ex. "oai-xxx-embedding-dev" or "text-embedding-3-small"
embed_Api_Version = 'your-api-version' #ex. "2024-02-01"

chat_Endpoint = "your-response-deployment-endpoint" # ex. https://aif-xxxxxx.services.ai.azure.com
chat_Name = "your-response-deployment-name" #ex. "oai-xxxx-chat-dev" or "gpt-40-mini"
chat_Api_Version = "your-api-version" #ex. "2024-02-01 

SetEmbeddingEndpointInfo(embed_Endpoint, embed_Deployment_Name, embed_Api_Version, api_Key)
SetCompletionEndpointInfo(chat_Endpoint, chat_Name, chat_Api_Version, api_Key)

# Open or create a persistent DB
OpenDatabase("compressDB", "./vectorDB")

# Ingest multiple documents
IngestFile("./ExampleFiles/Sample_TXT.txt", "compressDB")
IngestFile("./ExampleFiles/Sample_PDF.pdf", "compressDB")
IngestFile("./ExampleFiles/Sample_DOCX.docx", "compressDB")

# Ask a question that forces long-context compression
question = (
    "Provide a detailed summary that combines all major ideas across the documents. "
    "Include key differences, shared themes, and any conflicting information."
)

answer = GenerateAnswer(question)

print("\nQUESTION:")
print(question)
print("\nANSWER:")
print(answer)