# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:42:10 2026

#Sets up a temporary test database for unit tests and ensures it exists. 

@author: klusm
"""

import pytest
import os
import shutil

TEST_DB = "./test_vectorDB"

@pytest.fixture(scope="session", autouse=True)
def clean_test_db():
    if os.path.exists(TEST_DB):
        shutil.rmtree(TEST_DB)
    os.makedirs(TEST_DB, exist_ok=True)
    yield
    shutil.rmtree(TEST_DB)