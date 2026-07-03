# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:48:16 2026

Validates that ingestion of embedded text into the database is performing correctly. 

@author: klusm
"""

from ragpy.VectorDatabase import OpenDatabase, AddToDatabase, CheckIfInDatabase
from tests.MockAzureFunctions import FakeEmbed

def test_vector_database():
    OpenDatabase("TestDB", "./test_vectorDB")

    chunks = ["chunk one", "chunk two"]
    vectors = [FakeEmbed(c) for c in chunks]

    AddToDatabase(chunks, vectors, "test_source")

    assert CheckIfInDatabase("test_source") is True