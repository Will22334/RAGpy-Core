# -*- coding: utf-8 -*-
"""

File loading utilities for the RAGpy pipeline.

This module provides deterministic loaders for .txt, .pdf, and .docx
documents. Each loader extracts raw text content and returns it as a
single string suitable for downstream chunking and embedding. The
loading behavior is intentionally simple and predictable to support
both production usage and unit testing.

@author: klusm
"""
import os
from docx import Document
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

def LoadFile(path):
    """
    Load a .txt, .pdf, or .docx file and return its contents as a string.
    
    This function inspects the file extension and dispatches to the
    appropriate loader. Unsupported file types raise a ValueError.
    
    Args:
        path (str):
            Path to the file to load.
    
    Returns:
        str:
            The extracted text content from the file.
    
    Raises:
        FileNotFoundError:
            If the file does not exist.
        ValueError:
            If the file extension is not one of: .txt, .pdf, .docx.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        return _LoadText(path)
    elif ext == ".pdf":
        return _LoadPDF(path)
    elif ext == ".docx":
        return _LoadDOCX(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    
        
def _LoadText(path):
    """
    Load a UTF‑8 encoded plain text file.

    Args:
        path (str):
            Path to the .txt file.

    Returns:
        str:
            The full text content of the file.

    Raises:
        FileNotFoundError:
            If the file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _LoadPDF(path):
    """
    Extract text from a PDF file using pypdf.

    Each page is processed individually. Pages that contain no extractable
    text (common in scanned PDFs) are skipped.

    Args:
        path (str):
            Path to the .pdf file.

    Returns:
        str:
            Concatenated text extracted from all pages, separated by newlines.

    Raises:
        FileNotFoundError:
            If the file does not exist.
        Exception:
            If the PDF cannot be parsed or read.
    """
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def _LoadDOCX(path):
    """
    Extract text from a Microsoft Word .docx file.

    Args:
        path (str):
            Path to the .docx file.

    Returns:
        str:
            The concatenated text of all paragraphs in the document.
    """
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])
