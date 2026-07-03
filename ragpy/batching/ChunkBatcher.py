# -*- coding: utf-8 -*-
"""
Chunk batching utilities for the RAGpy pipeline.

This module provides simple, deterministic batching logic used during
embedding. Text chunks are grouped into fixed‑size batches to reduce
API calls and improve throughput. The batching behavior is intentionally
minimal and predictable to support both production usage and unit testing.

Batching preserves the original chunk order and ensures that embeddings
returned from Azure OpenAI can be mapped back to their source chunks
without ambiguity.
"""

def BatchChunks(chunks, batch_size=16):
    """
    Group text chunks into batches for embedding.
    
    This function yields sequential batches of text chunks, each containing
    up to `batch_size` items. Batching reduces the number of embedding
    requests sent to Azure OpenAI and improves throughput while preserving
    deterministic ordering.
    
    Args:
        chunks (list[str]):
            A list of text chunks produced by the chunker.
        batch_size (int):
            Maximum number of chunks per batch. Must be >= 1.
    
    Yields:
        list[str]:
            A batch of text chunks in their original order.
    
    Raises:
        ValueError:
            If `batch_size` is less than 1.
    
    Notes:
        - Empty input returns immediately without yielding any batches.
        - Ordering is preserved exactly as provided.
        - The caller may convert the generator to a list if needed:
              batches = list(BatchChunks(chunks))
        - The function is intentionally minimal to support monkeypatching
          and deterministic unit tests.
    """
    #Safety for empty input
    if not chunks:
        return
    
    #Break the chunks into batches of size batch_size 
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]
        
def _BatchDebugger(batched_chunks):
    """
    Debug utility for inspecting batched chunk structures.

    This helper validates that the provided object is a list of batches
    and prints structural information useful during development, including
    batch count and batch sizes.

    Args:
        batched_chunks (list):
            A list of batches produced by BatchChunks.

    Raises:
        TypeError:
            If `batched_chunks` is not a list.
    """
    if not isinstance(batched_chunks, list):
        raise TypeError("Embedding requires a list of batches, not a generator or whatever else is being passed.")
        
    print("Type:", type(batched_chunks))
    print("Is list:", isinstance(batched_chunks, list))
    print("Length:", len(batched_chunks))
    print("First batch type:", type(batched_chunks[0]))
    print("First batch length:", len(batched_chunks[0]))
    