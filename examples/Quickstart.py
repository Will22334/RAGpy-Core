# -*- coding: utf-8 -*-
"""
Basic Ingestion and Querying
"""

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

# Open or create a persistent ChromaDB collection
OpenDatabase("testDB", "./vectorDB")

# Ingest a document
IngestFile("./ExampleFiles/Sample_TXT.txt", "testDB")

# Ask a question
answer = GenerateAnswer("What is RAG?")
print(answer)