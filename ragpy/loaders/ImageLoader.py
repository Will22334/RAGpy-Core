# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:36:00 2026

This module is part of the RAGpy loaders subsystem. It provides safe and
validated loading of image files using Pillow. The loader ensures that
invalid, corrupted, or inaccessible image files fail with clear and
descriptive errors so that the ingestion pipeline can handle them
gracefully.

@author: klusm
"""

from PIL import Image
import os

def OpenImage(path):
    
    """
    Load an image from disk using Pillow.

    Parameters
    ----------
    path : str
        The filesystem path to the image file.

    Returns
    -------
    Image.Image
    A Pillow Image object representing the loaded image.

    Raises
    ------
    ValueError
        If the path is missing, invalid, or the file is not a valid image.
    
    FileNotFoundError
        If the file does not exist at the given path.

    PermissionError
        If the file cannot be accessed due to permission restrictions.
    
    RuntimeError
        For any unexpected error encountered while loading the image.

    Notes
    -----
    This function forces the image to fully load using `im.load()`
    to ensure that corrupted or partially readable files fail early.
    """
    
    # Check if path is provided
    if not path or not isinstance(path, str):
        raise ValueError("OpenImage: 'path' must be a valid string.")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"OpenImage: File not found at path '{path}'.")
    
    try: 
        
        im = Image.open(path)
        im.load()
        return im
    
    except Image.UnidentifiedImageError:
        raise ValueError(f"OpenImage: File at '{path}' is not a valid image format.")

    except PermissionError:
        raise PermissionError(f"OpenImage: Permission denied when accessing '{path}'.")

    except Exception as e:
        raise RuntimeError(f"OpenImage: Unexpected error while loading '{path}': {e}")
        