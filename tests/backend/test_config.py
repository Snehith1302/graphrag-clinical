"""
Tests for backend configuration loading.
"""
from backend.app.config import settings

def test_settings_load():
    """Verifies settings load environment variables successfully."""
    assert settings.EMBEDDING_MODEL_NAME is not None
    assert settings.MAX_HOP_DEPTH > 0
    assert settings.VECTOR_TOP_K > 0
