# -*- coding: utf-8 -*-
"""
Tests for the ImageLoader module in loaders/.
"""

import pytest
from PIL import Image
from ragpy.loaders.ImageLoader import OpenImage

def test_openimage_valid(tmp_path):
    # Create a temporary valid image file
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    loaded = OpenImage(str(img_path))
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (10, 10)

def test_openimage_missing_file():
    with pytest.raises(FileNotFoundError):
        OpenImage("nonexistent_image.png")

def test_openimage_invalid_path_type():
    with pytest.raises(ValueError):
        OpenImage(None)

def test_openimage_invalid_image(tmp_path):
    # Create a non-image file
    bad_path = tmp_path / "not_image.txt"
    bad_path.write_text("This is not an image.")

    with pytest.raises(ValueError):
        OpenImage(str(bad_path))
