# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:49:43 2026

Tests retrieval from the database by embedding some example chunks of text of related topics and adding them to the test database,
it then queries the database with example text which is related. the database should return the two topics.  

@author: klusm
"""

from ragpy.VectorDatabase import OpenDatabase, AddToDatabase
from ragpy.DatabaseRetriever import RetrieveTopK
from ragpy.mocks.MockAzureFunctions import FakeEmbed

def test_retrieval():
    OpenDatabase("TestDB", "./test_vectorDB")

    chunks = ["engine vibration analysis", "flight control systems"]
    vectors = [FakeEmbed(c) for c in chunks]

    AddToDatabase(chunks, vectors, "test_source")

    query_vec = FakeEmbed("vibration")
    results = RetrieveTopK(query_vec, k=2)

    assert len(results) == 2
    assert "vibration" in results[0]["text"]
