# -*- coding: utf-8 -*-
"""
Azure OpenAI relay utilities for the RAGpy pipeline.

This module provides a thin, deterministic wrapper around the Azure OpenAI
Python SDK. It exposes simple embedding and chat‑completion functions used
throughout the RAG workflow, while keeping configuration isolated and
monkeypatch‑friendly for unit testing.

Two independent clients are supported:
    • Embedding client — for generating text embeddings.
    • Chat completion client — for LLM‑based scoring, compression, and
      answer generation.

The relay is intentionally minimal to ensure predictable behavior and easy
replacement during offline tests.

"""
from openai import AzureOpenAI

#new Azure OpenAI SDK
embedding_Client = None
chat_Client = None
embedding_Deployment = None
completion_Deployment = None

def SetCompletionEndpointInfo(completion_Endpoint, completion_Deployment_Name, completion_Api_Version, api_Key):
    """
    Configure the Azure OpenAI chat completion client.

    This sets up the client used for all LLM‑based operations in the RAG
    pipeline, including reranking, compression, and final answer generation.

    Args:
        completion_Endpoint (str):
            The Azure endpoint URL for chat completions.
        completion_Deployment_Name (str):
            The name of the deployed chat model.
        completion_Api_Version (str):
            The API version to use.
        api_Key (str):
            The Azure OpenAI API key.

    Returns:
        None
    """
    global chat_Client, completion_Deployment
    completion_Deployment = completion_Deployment_Name

    chat_Client = AzureOpenAI(
        azure_endpoint=completion_Endpoint,
        api_version=completion_Api_Version,
        api_key=api_Key
            
    )
    
    return

def SetEmbeddingEndpointInfo(embedding_Endpoint, embedding_Deployment_Name, embedding_Api_Version, azure_Api_Key):
    """
    Configure the Azure OpenAI embedding client.

    This sets up the client used for generating embeddings during ingestion
    and query processing.

    Args:
        embedding_Endpoint (str):
            The Azure endpoint URL for embeddings.
        embedding_Deployment_Name (str):
            The name of the deployed embedding model.
        embedding_Api_Version (str):
            The API version to use.
        azure_Api_Key (str):
            The Azure OpenAI API key.

    Returns:
        None
    """
    global embedding_Client, embedding_Deployment
    embedding_Deployment = embedding_Deployment_Name
    
    embedding_Client = AzureOpenAI(
        azure_endpoint=embedding_Endpoint,
        api_version=embedding_Api_Version,
        api_key = azure_Api_Key
        )
    
    return
    
def EmbedText(text):
    """
    Generate an embedding vector for a single text string.

    Args:
        text (str):
            The input text to embed.

    Returns:
        list[float]:
            A 3072‑dimensional embedding vector produced by the configured
            Azure OpenAI embedding model.

    Raises:
        ValueError:
            If the returned embedding dimension is unexpected.

    Notes:
        - This function is intentionally simple to support monkeypatching
          during unit tests.
        - The embedding client must be configured before calling this
          function.
    """
    vector = embedding_Client.embeddings.create(
        model=embedding_Deployment,
        input=text
    ).data[0].embedding

    if len(vector) != 3072:
        raise ValueError(f"Unexpected embedding dimension: {len(vector)}")

    return vector 

def EmbedChunksInBatches(batched_chunks, batch_size=16):
    """
    Embed multiple batches of text chunks.

    Each batch is sent to the embedding model as a single request, and all
    resulting vectors are returned in a flat list (one embedding per chunk).

    Args:
        batched_chunks (list[list[str]]):
            A list of batches, each containing text chunks.
        batch_size (int):
            Optional batch size hint (unused but kept for compatibility).

    Returns:
        list[list[float]]:
            A flat list of embedding vectors, one per chunk.

    Notes:
        - Ordering is preserved: embeddings appear in the same order as
          the input chunks.
        - This function is intentionally simple to support monkeypatching.
    """
    all_vectors = []

    for batch in batched_chunks:
        response = embedding_Client.embeddings.create(
            model=embedding_Deployment,
            input=batch
        )

        for item in response.data:
            all_vectors.append(item.embedding)

    return all_vectors

def ChatCompletion(prompt):
    """
    Generate a chat completion response using the configured Azure OpenAI client.

    Args:
        prompt (str):
            The user prompt to send to the model.

    Returns:
        str:
            The model's response text. Returns an empty string if no chat
            client is configured.

    Notes:
        - This function is intentionally simple to support monkeypatching
          during unit tests.
        - No additional formatting or metadata is returned.
    """
    client = GetChatClient()
    if client is None:
        # In tests, FakeChat will override this entirely.
        return ""

    response = client.chat.completions.create(
        model=GetChatModel(),
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def GetChatClient():
    """
    Retrieve the configured Azure OpenAI chat client.

    Returns:
        AzureOpenAI | None:
            The chat client instance, or None if not configured.
    """
    return chat_Client

def GetChatModel():
    """
    Retrieve the configured chat model deployment name.

    Returns:
        str | None:
            The model deployment name, or None if not configured.
    """
    return completion_Deployment