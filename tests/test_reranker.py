# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:53:01 2026

Creates a list of text with the topic in index 1, then uses the reranker to see if that topic related to the query
moves to index 0. 

@author: klusm
"""

from ragpy.Reranker import Rerank
from ragpy.mocks.MockAzureFunctions import FakeReranker

def test_reranker(monkeypatch):
    chunks = [
        {"text": "engine vibration analysis"},
        {"text": "flight control systems"}
    ]

    monkeypatch.setattr("ragpy.Reranker.Rerank", FakeReranker)

    reranked = Rerank("vibration", chunks)

    assert reranked[0]["text"] == "flight control systems"
