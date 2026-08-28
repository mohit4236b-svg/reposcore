chunk1 = '''"""
Tests for model loading, prediction, and FastAPI endpoints.

These tests verify:
1. The ML model loads correctly from its saved file
2. A single prediction call returns a valid score/confidence in expected range
3. FastAPI endpoints return expected status codes and response shapes
"""
import os
import sys
import pytest
import joblib
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
'''