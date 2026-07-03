# -*- coding: utf-8 -*-
"""
Text chunking utilities for the RAGpy pipeline.

This module provides a deterministic sliding‑window chunker used to split
raw text into fixed‑size overlapping segments. Chunking is a foundational
step in the RAG workflow, enabling efficient embedding, retrieval, and
context construction. The design is intentionally simple and predictable
to support both production usage and unit testing.

Sweet Spot for RAG:
    800–1,200 tokens per chunk
    100–200 token overlap

@author: klusm
"""
import logging

logger = logging.getLogger(__name__)


def TextToChunk(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """
    Split text into fixed‑size overlapping chunks using a sliding window.

    This function generates sequential chunks of up to `chunk_size`
    characters, with `overlap` characters of backward context preserved
    between consecutive chunks. Overlap helps maintain semantic continuity
    across chunk boundaries, improving downstream retrieval and reranking.

    Args:
        text (str):
            The full text to chunk.
        chunk_size (int):
            Maximum number of characters per chunk.
        overlap (int):
            Number of characters to overlap between consecutive chunks.

    Returns:
        list[str]:
            A list of text chunks. Empty or whitespace‑only input returns
            an empty list.

    Notes:
        - Chunking is based purely on character length for simplicity and
          deterministic behavior.
        - Overlap ensures important context is not lost between chunks.
        - Trailing empty or whitespace‑only chunks are removed.
        - Ordering is preserved exactly as the text appears.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move back slightly for context overlap
    
    #Remove empty spaces
    chunks = [c for c in chunks if c and c.strip()]
    
    logger.info(f"Text was broken into {len(chunks)} chunks.")
    
    return chunks