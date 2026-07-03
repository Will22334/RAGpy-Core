# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:58:07 2026

This text performs the entire RAGpy pipeline, opening the database, creating a test file with text and ingesting that
text into the database. It then queriest the database and ensures that an answer is created. 

@author: klusm
"""

from ragpy.RAGOrchestrator import IngestFile, GenerateAnswer
from ragpy.VectorDatabase import OpenDatabase
from tests.MockAzureFunctions import FakeEmbed, FakeEmbedChunks, FakeChat, FakeCompress, FakeReranker
from ragpy import AzureOpenAIRelay as AI
from ragpy import Reranker as RR
from ragpy import ChunkCompressor as CC

def test_orchestrator(monkeypatch):
    OpenDatabase("TestDB", "./test_vectorDB")

    # Create a small test file
    with open("test_doc.txt", "w") as f:
        f.write("Engine vibration analysis is essential for aerospace safety.")

    # Mock Azure + LLM components
    monkeypatch.setattr(AI, "EmbedText", FakeEmbed)
    monkeypatch.setattr(AI, "EmbedChunksInBatches", FakeEmbedChunks)
    monkeypatch.setattr(AI, "ChatCompletion", FakeChat)
    monkeypatch.setattr(RR, "Rerank", FakeReranker)
    monkeypatch.setattr(CC, "CompressChunks", FakeCompress)

    IngestFile("test_doc.txt", "TestDB")

    answer = GenerateAnswer("What is vibration analysis?", "TestDB")

    assert answer == "Mocked answer for testing."