# -*- coding: utf-8 -*-
"""
LLM-based context compressor for the RAGpy pipeline.

This module provides a lightweight, deterministic interface for reducing
multiple retrieved chunks into a single compact summary. The compressor
uses AzureOpenAIRelay to generate a concise, query‑aware context block
that preserves meaning while reducing token usage. The design is
intentionally simple to support both production usage and easy
monkeypatching during unit tests.

"""
from ragpy import AzureOpenAIRelay as AI

def CompressChunks(query, chunks):
    """
    Compress a list of retrieved chunks into a shorter, unified context block.

    This function sends the provided chunks and user query to an LLM,
    requesting a concise summary that preserves the information most
    relevant to the query. It is primarily used to reduce token usage
    during prompt construction while maintaining grounding in the
    retrieved content.

    Args:
        query (str):
            The user query that determines what information is relevant.
        chunks (list[dict]):
            A list of chunk objects, each containing at least a "chunk"
            field (or "text") holding the raw text to be compressed.

    Returns:
        str:
            A compressed context string. Returns an empty string if the
            compression model is not configured or returns no output.

    Notes:
        - The function is intentionally simple to allow monkeypatching
          during unit tests.
        - The compression prompt is defined inline for clarity and
          isolation.
        - If AzureOpenAIRelay is not configured, the function safely
          returns an empty string.
    """
    # If monkeypatched, FakeCompress will run instead of this function.
    if AI.GetChatClient() is None:
        return ""

    text = "\n\n".join(c.get("chunk") or c.get("text", "") for c in chunks)

    prompt = f"""
    You are a compression model. Your job is to compress the following text
    into a concise summary that preserves meaning and focuses on information
    relevant to the user query:

    Query:
    {query}

    {text}

    Return ONLY the compressed summary.
    """

    return AI.ChatCompletion(prompt)