# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:11:39 2026

Allows the ability of local install using : pip install -e .

@author: klusm
"""
from setuptools import setup, find_packages

setup(
    name="ragpy",
    version="0.1.0",
    description="A modular RAG (Retrieval-Augmented Generation) library for Python.",
    author="William Klusman",
    packages=find_packages(),
    install_requires=[
        "chromadb>=0.4.0",
        "numpy>=1.20",
        "tiktoken>=0.5.0",
        "requests>=2.0",
    ],
    python_requires=">=3.9",
)