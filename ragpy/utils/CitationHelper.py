# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:23:56 2026

@author: klusm
"""

def ResolveCitations(answer, retrieved_chunks):
    """
    Replace [CITATION X] markers in the model's answer with
    human-readable source references.
    """
    for i, r in enumerate(retrieved_chunks):
        source = r["metadata"].get("source", "unknown")
        chunk_id = r["metadata"].get("chunk", "N/A")

        resolved = f"({source}, chunk {chunk_id})"
        answer = answer.replace(f"[CITATION {i}]", resolved)

    return answer


def BuildReferences(retrieved_chunks):
    """
    Build a reference list for the bottom of the final answer.
    """
    refs = []
    for i, r in enumerate(retrieved_chunks):
        source = r["metadata"].get("source", "unknown")
        chunk_id = r["metadata"].get("chunk", "N/A")
        refs.append(f"{i}. {source} (chunk {chunk_id})")
    return "\n".join(refs)