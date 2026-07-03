# -*- coding: utf-8 -*-
"""
Retrieval Layer for RAG
Created on Tue Jun 2 2026
@author: William
"""

from ragpy import VectorDatabase as VD
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def RetrieveTopK(embedded_query: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top‑K most relevant chunks from the active ChromaDB collection.

    This function performs a vector similarity search against the currently
    opened database collection using the provided query embedding. Results
    are normalized into a consistent structure expected by downstream
    components such as the reranker, compressor, and orchestrator.

    Args:
        embedded_query (list[float]):
            The embedding vector representing the user query. Must match the
            dimensionality of the embeddings stored in the database.
        k (int):
            Number of chunks to retrieve from the vector database.

    Returns:
        list[dict]:
            A list of retrieved chunk objects, each containing:
                - "text": The raw chunk text stored in the database.
                - "metadata": Metadata associated with the chunk, including:
                    • "source": The originating filename.
                    • "chunk_id": The index of the chunk within that file.
                - "distance": The vector distance score returned by ChromaDB.
                  Lower values indicate closer matches.

    Raises:
        RuntimeError:
            If no database collection is currently open.

    Notes:
        - Distance values are returned directly from ChromaDB. If the backend
          does not provide distances, they default to 0.0.
        - Output format is intentionally normalized to ensure compatibility
          with Reranker, ChunkCompressor, and RAGOrchestrator.
    """
    if VD.collection is None:
        raise RuntimeError("No database open. Call CreateDatabase() or OpenDatabase() first.")
    
    if len(embedded_query) != 3072:
        raise ValueError("Query embedding dimension mismatch.")

    results = VD.collection.query(
        query_embeddings=[embedded_query],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[0.0] * len(docs)])[0]
    logger.debug(f"Distance range: {min(dists)} to {max(dists)}")

    normalized = []
    for doc, meta, dist in zip(docs, metas, dists):
        normalized.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
            
        })
    
    logger.info(f"Retrieved {len(normalized)} chunks from database.")
    
    return normalized