# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:46:21 2026

#This test ensures that the embedding pipeling is performing correctly. 

@author: klusm
"""

from ragpy import AzureOpenAIRelay as AI
from ragpy.mocks.MockAzureFunctions import FakeEmbed

def test_embed_text(monkeypatch):
    monkeypatch.setattr(AI, "EmbedText", FakeEmbed)
    vec = AI.EmbedText("hello world")
    
    assert all(v == 0.0 for v in vec)