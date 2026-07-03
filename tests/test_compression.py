# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:56:24 2026

This ensures that the compression pipeline can return the proper context. 

@author: klusm
"""

from ragpy import ChunkCompressor as CC
from tests.MockAzureFunctions import FakeCompress

def test_compression(monkeypatch):
    chunks = [
        {"chunk": "Engine vibration analysis is critical."},
        {"chunk": "Flight control systems manage stability."}
    ]

    monkeypatch.setattr("ragpy.ChunkCompressor.CompressChunks", FakeCompress)

    compressed = CC.CompressChunks("vibration", chunks)

    assert compressed == "compressed context for testing"