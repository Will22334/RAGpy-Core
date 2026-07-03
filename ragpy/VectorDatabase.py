# -*- coding: utf-8 -*-
# Global state:
# - `client` holds the active ChromaDB PersistentClient
# - `collection` holds the active ChromaDB collection
"""
ChromaDB persistence utilities for the RAGpy pipeline.

This module provides a thin wrapper around ChromaDB's PersistentClient,
offering simple functions for creating, opening, and populating a vector
database. The design is intentionally minimal and deterministic to support
production usage and easy monkeypatching during unit tests.

Global State:
    client     — the active ChromaDB PersistentClient
    collection — the active ChromaDB collection
"""
import chromadb
import os
import logging

client = None 
collection = None
logger = logging.getLogger(__name__)


def CreateDatabase(dbName, path="./vectorDB"):
    """
    Create a new persistent ChromaDB collection.

    Initializes a PersistentClient at the given path and unconditionally
    creates a new collection with the specified name. If the collection
    already exists, ChromaDB will raise an error.

    Args:
        dbName (str):
            Name of the collection to create.
        path (str):
            Filesystem path for the persistent database.

    Returns:
        chromadb.api.models.Collection.Collection:
            The newly created collection.

    Side Effects:
        - Sets the global `client` and `collection` variables.
    """
    global client, collection
    
    client = chromadb.PersistentClient(path)
   
    collection = client.create_collection(dbName)

    return collection


def OpenDatabase(dbName, path="./vectorDB"):
    """
    Open an existing ChromaDB collection, creating it if necessary.

    Initializes a PersistentClient at the given path and attempts to load
    an existing collection. If the collection does not exist, it is created
    automatically.

    Args:
        dbName (str):
            Name of the collection to open.
        path (str):
            Filesystem path for the persistent database.

    Returns:
        chromadb.api.models.Collection.Collection:
            The opened or newly created collection.

    Side Effects:
        - Sets the global `client` and `collection` variables.
    """
    global client, collection
    
    client = chromadb.PersistentClient(path)
    
    try:
        collection = client.get_collection(dbName)
    except chromadb.errors.NotFoundError:
        logger.info("Database does not exist. Creating...")
        collection = CreateDatabase(dbName, path)
    
    return collection

def AddToDatabase(chunks, vectors, source_name):
    """
    Add embedded text chunks to the active ChromaDB collection.

    Stores text chunks, their embedding vectors, and associated metadata
    in the currently opened collection. Each chunk receives a globally
    unique ID based on the source filename and its index.

    Args:
        chunks (list[str]):
            Text chunks extracted from a document.
        vectors (list[list[float]]):
            Embedding vectors corresponding to each chunk.
        source_name (str):
            The name of the source document (e.g., filename).

    Returns:
        None

    Raises:
        RuntimeError:
            If no database collection is currently open.
        ValueError:
            If chunk/vector counts differ or an embedding has an unexpected
            dimensionality.

    Notes:
        - Embedding vectors are validated to ensure a 3072‑dimensional shape.
        - Metadata stores only lightweight identifiers (`source`, `chunk_id`)
          to avoid duplicating chunk text.
    """
    global collection
    if collection is None:
        raise RuntimeError("No database open. Call CreateDatabase() or OpenDatabase() first.")
        
    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must have the same length.") 
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        
        if len(vector) != 3072:
            raise ValueError(f"Invalid embedding length: {len(vector)}")

        ids.append(f"{source_name}_{i}")
        embeddings.append(vector)
        documents.append(chunk)
        metadatas.append({"source": source_name, "chunk_id": i})

    collection.add(
        ids=ids,
        embeddings=embeddings,

        documents=documents,
        metadatas=metadatas
    )
    
def CheckIfInDatabase(filepath):
    """
    Determine whether a file has already been ingested into the database.

    Uses metadata filtering to check whether any stored chunk originates
    from the given file.

    Args:
        filepath (str):
            Path to the file being checked.

    Returns:
        bool:
            True if any metadata entry references the file, False otherwise.

    Raises:
        RuntimeError:
            If no database collection is currently open.

    Notes:
        - Matching is performed using the filename only (basename).
        - Uses ChromaDB's `where` filtering for efficient lookup.
    """
    if collection is None:
       raise RuntimeError("No database open. Call CreateDatabase() or OpenDatabase() first.")

    filename = os.path.basename(filepath)

    results = collection.get(where={"source": filename})
    
    for m in results["metadatas"]:
        if m["source"] == filename:
            return True

    return False