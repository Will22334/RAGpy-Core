# -*- coding: utf-8 -*-
"""
LLM-based and deterministic rerankers for RAG pipelines.

This module provides two reranking strategies:

    • RerankLLM — the primary reranker used during real RAGpy execution.
      It evaluates chunk relevance using an LLM scoring prompt and returns
      chunks sorted by descending relevance score while preserving metadata.

    • Rerank — a deterministic, test-friendly fallback used exclusively
      for unit testing. It sorts chunks by text length to ensure stable,
      reproducible ordering without requiring an LLM.

Both functions accept retrieved chunk objects produced by the retrieval
layer and return normalized dictionaries suitable for downstream
compression and prompt construction.
"""

from ragpy import AzureOpenAIRelay as AI

def RerankLLM(query, retrieved_chunks):
    """
    Rerank retrieved chunks using an LLM-based scoring model.

    Each chunk is evaluated for relevance to the user query by sending a
    scoring prompt to the LLM via AzureOpenAIRelay. The model is expected
    to return a floating‑point number between 0 and 1. Invalid or malformed
    output safely defaults to 0.0. Scores are clamped to the range [0, 1].

    Args:
        query (str):
            The user query used to evaluate relevance.
        retrieved_chunks (list[dict]):
            A list of retrieved chunk objects containing:
                - "text": The chunk text.
                - "metadata": Metadata including "source" and "chunk_id".
                - "distance": The vector distance score.

    Returns:
        list[dict]:
            A list of dictionaries with keys:
                - "text": The chunk text.
                - "metadata": Original metadata preserved.
                - "score": The relevance score (float).
            Sorted in descending score order.

    Notes:
        - This function is used during real RAG pipeline execution.
        - It is intentionally monkeypatch‑friendly for offline testing.
        - Metadata is preserved to support compression and citation workflows.
    """
    scored = []

    for r in retrieved_chunks:
        chunk_text = r.get("chunk") or r.get("text")

        prompt = f"""
        You are a scoring model. Rate the relevance of the chunk to the query from 0 to 1.
        Return ONLY a number.
        
        Query:
        {query}
        
        Chunk:
        {chunk_text}
        """

        score_text = AI.ChatCompletion(prompt).strip()

        try:
            score = float(score_text)
            score = max(0.0, min(score, 1.0))
        except:
            score = 0.0

        scored.append({
            "text": chunk_text,
            "metadata": r.get("metadata", {}),
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def Rerank(query, retrieved_chunks):
    """
Deterministic, test-friendly reranker used only for unit testing.

This fallback reranker provides predictable behavior without relying
on an LLM. It normalizes chunk objects and sorts them by text length,
placing shorter chunks first. All metadata is preserved.

Args:
    query (str):
        The user query (unused in this heuristic but included for API consistency).
    retrieved_chunks (list[dict]):
        A list of retrieved chunk objects containing "text" and "metadata".

Returns:
    list[dict]:
        A list of normalized chunk dictionaries sorted by ascending text length.

Notes:
    - This function is used exclusively for testing.
    - Tests expect shorter chunks to appear first.
    - Metadata is preserved to maintain compatibility with downstream steps.
"""
    normalized = []
    for r in retrieved_chunks:
        text = r.get("text") or r.get("chunk") or ""
        normalized.append({
            "text": text,
            **{k: v for k, v in r.items() if k not in ["text", "chunk"]}
        })

    # Heuristic: shorter text is usually more focused → tests expect this ordering.
    return sorted(normalized, key=lambda c: len(c["text"]))