# -*- coding: utf-8 -*-
"""
RAGpy Core Package
Provides a lightweight, modular RAG pipeline for Azure OpenAI + ChromaDB.
"""

# Azure OpenAI relay
from .AzureOpenAIRelay import (
    SetEmbeddingEndpointInfo,
    SetCompletionEndpointInfo,
)

# Orchestrator
from .RAGOrchestrator import (
    IngestFile,
    GenerateAnswer,
    BuildContext,
    BuildPrompt,
)

# Vector database
from .VectorDatabase import (
    OpenDatabase,
    AddToDatabase,
    CheckIfInDatabase,
)

# Utilities
from .utils.CitationHelper import (
    ResolveCitations,
    BuildReferences,
)

__all__ = [
    "SetEmbeddingEndpointInfo",
    "SetCompletionEndpointInfo",
    "IngestFile",
    "GenerateAnswer",
    "BuildContext",
    "BuildPrompt",
    "OpenDatabase",
    "AddToDatabase",
    "CheckIfInDatabase",
    "ResolveCitations",
    "BuildReferences",
]