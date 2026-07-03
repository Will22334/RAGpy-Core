# -*- coding: utf-8 -*-
"""
RAG Orchestrator for the RAGpy pipeline.

This module coordinates the full Retrieval‑Augmented Generation workflow,
including file ingestion, chunking, batching, embedding, retrieval,
reranking, context construction, prompt building, and final answer
generation. It acts as the high‑level interface that ties together all
RAGpy components:

    • VectorDatabase — persistent storage of embedded chunks
    • FileLoader — file loading and text extraction
    • TextChunker — chunking of raw text
    • ChunkBatcher — batching for efficient embedding
    • AzureOpenAIRelay — embedding and LLM completion
    • DatabaseRetriever — vector search
    • Reranker — LLM‑based and deterministic reranking
    • ChunkCompressor — context compression for efficient prompting

All functions in this module operate at the orchestration level and are
intended to be used directly by applications integrating RAGpy.
"""
import os
import logging

from .VectorDatabase import OpenDatabase, CheckIfInDatabase, AddToDatabase
from .loaders.FileLoader import LoadFile
from .loaders.TextChunker import TextToChunk
from .batching.ChunkBatcher import BatchChunks

import ragpy.AzureOpenAIRelay as AI
import ragpy.Reranker as RR
import ragpy.DatabaseRetriever as DR
import ragpy.ChunkCompressor as CC

logger = logging.getLogger(__name__)

def IngestFile(path, databaseName, databaseLoc = "./vectorDB"):
    """
    Ingest a file into the vector database.

    Performs the full ingestion pipeline:
        1. Load the file from disk.
        2. Chunk the extracted text.
        3. Batch chunks for efficient embedding.
        4. Embed all batches using AzureOpenAIRelay.
        5. Store embedded chunks in the persistent vector database.

    If the file has already been ingested, the operation is skipped.

    Args:
        path (str):
            Path to the file to ingest.
        databaseName (str):
            Name of the vector database collection.
        databaseLoc (str):
            Directory where the database is stored.

    Returns:
        None
    """
    filename = os.path.basename(path)
    logger.info(f"Found the file: {filename}")
    
    #Open the Database
    OpenDatabase(databaseName, databaseLoc)
    
    # Check if already embedded
    if CheckIfInDatabase(path):
        logger.info(f"Skipping {filename}: already embedded.")
        return
    else:
        logger.info("File did not exist in DB, Embedding.")
        text = LoadFile(path)
        chunks = TextToChunk(text)
        batches = list(BatchChunks(chunks))
        vectors = AI.EmbedChunksInBatches(batches)

    AddToDatabase(chunks, vectors, filename)
    
    logger.info(f"Index operation complete. {filename} was split into {len(chunks)} chunks, batched into {len(batches)} batches, and indexed.")

def BuildContext(retrieved_chunks):
    """
    Construct a readable context block from retrieved chunks.

    Each chunk is formatted with a citation marker and includes metadata
    such as the source filename and chunk index. This context block is
    used directly in RAG prompt construction.

    Args:
        retrieved_chunks (list[dict]):
            Retrieved chunk objects containing:
                - "text": The chunk text.
                - "metadata": Metadata including "source" and "chunk_id".

    Returns:
        str:
            A formatted context block with citation markers.
    """
    context = []

    for i, r in enumerate(retrieved_chunks):
        source = r["metadata"].get("source", "unknown")
        chunk_id = r["metadata"].get("chunk_id", "N/A")
        text = r.get("text")

        context.append(f"[CITATION {i}] Source: {source} (chunk {chunk_id})\n{text}\n")

    return "\n".join(context)

def BuildPrompt(context, query, use_citations=True):
    """
    Build a full RAG prompt including context, optional citation instructions,
    and the user query.

    Args:
        context (str):
            The context block to include in the prompt.
        query (str):
            The user question.
        use_citations (bool):
            Whether to include citation instructions for the LLM.

    Returns:
        str:
            The constructed prompt ready for LLM completion.
    """
    if use_citations:
        citation_instructions = """
        When you use information from a chunk, cite it like this: [CITATION X].
        Do not invent citations. Only cite chunks that appear in the context.
        """
    else:
        citation_instructions = ""
    
    prompt = f"""
    {citation_instructions}

    ### Context:
    {context}
    
    ### Focus specifically on information relevant to the user query:
    {query}
    
    ### Your Answer (friendly, readable, grounded):
    """
    return prompt

def GenerateAnswer(query, k=5):
    """
    Generate a grounded answer using the full RAG pipeline.

    Steps:
        1. Embed the user query.
        2. Retrieve top‑K candidate chunks from the vector database.
        3. Rerank candidates using the LLM-based reranker.
        4. Compress the top chunks into a compact context summary.
        5. Build a RAG prompt using the compressed context.
        6. Generate the final answer using AzureOpenAIRelay.

    Args:
        query (str):
            The user question.
        k (int):
            Number of chunks to retrieve before reranking.

    Returns:
        str:
            The final grounded answer generated by the LLM.

    Raises:
        ValueError:
            If the query embedding has an unexpected dimensionality.
    """
    embedded_Query = AI.EmbedText(query)
    
    if len(embedded_Query) != 3072:
        raise ValueError("Query embedding dimension mismatch.")

    # Step 1: Retrieve more chunks
    retrieved = DR.RetrieveTopK(embedded_Query, k)

    # Step 2: Rerank them
    reranked = RR.RerankLLM(query, retrieved)

    # Step 3: Keep top chunks for compression
    top_for_compression = [
    {"chunk": item["text"], "metadata": item["metadata"]}
    for item in reranked[:10]
    ]
    
    # Step 4: Compress them
    compressed_context = CC.CompressChunks(query, top_for_compression)

    # Step 5: Build prompt using compressed context
    prompt = BuildPrompt(compressed_context, query, use_citations=False)

    # Step 6: Generate answer
    raw_answer = AI.ChatCompletion(prompt)

    return raw_answer
