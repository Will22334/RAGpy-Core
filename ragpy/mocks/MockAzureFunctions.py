# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:06:40 2026

#Fake functions to bypass Azure calls during testing. 

@author: klusm
"""

def FakeEmbed(text):
    # Deterministic fake embedding
    return [0.0] * 3072

def FakeEmbedChunks(chunks):
    return [[0.0] * 3072 for _ in chunks]

def FakeChat(prompt):
    return "Mocked answer for testing."

def FakeReranker(query, chunks):
    # Sort by text length for deterministic behavior
    return sorted(chunks, key=lambda c: len(c["text"]))

def FakeCompress(query, chunks):
    return "compressed context for testing"