"""
Unit tests for the Vector RAG retrieval baseline pipeline.
Mocks the SentenceTransformer embedding model at the module level for speed and offline stability.
"""
import os
import shutil
# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from unittest.mock import MagicMock

# 1. Mock SentenceTransformer globally BEFORE importing any retrieval module to prevent HF downloads
mock_model_instance = MagicMock()

def fake_encode(texts, **kwargs):
    # Returns simple mock unit vectors of dimension 384
    n_texts = len(texts)
    arr = np.zeros((n_texts, 384), dtype=np.float32)
    for idx, text in enumerate(texts):
        char_sum = sum(ord(c) for c in text)
        arr[idx, char_sum % 384] = 1.0
    return arr

mock_model_instance.encode.side_effect = fake_encode

# Patch the class in sentence_transformers module and vector_store namespace
# pyrefly: ignore [missing-import]
import sentence_transformers
import backend.app.retrieval.vector_store

mock_sent_transformer = MagicMock(return_value=mock_model_instance)
sentence_transformers.SentenceTransformer = mock_sent_transformer
backend.app.retrieval.vector_store.SentenceTransformer = mock_sent_transformer

# 2. Now import retrieval modules safely
from backend.app.models.schemas import Chunk
from backend.app.retrieval.vector_store import vector_store
from backend.app.retrieval.vector_retrieve import vector_retrieve

@pytest.fixture(autouse=True)
def setup_clean_vector_store():
    # Make sure we use a temporary directory for test index files
    test_index_dir = "./data/test_vector_index"
    vector_store.index_dir = test_index_dir
    vector_store.index_file = os.path.join(test_index_dir, "index.faiss")
    vector_store.metadata_file = os.path.join(test_index_dir, "chunks_metadata.json")
    
    # Initialize empty index
    vector_store.clear()
    
    yield
    
    # Clean up files after run
    if os.path.exists(test_index_dir):
        shutil.rmtree(test_index_dir, ignore_errors=True)

def test_add_and_search_chunks():
    # Add mock chunks
    chunks = [
        Chunk(chunk_id="c1", document_id="doc1", text="Metformin treats type 2 diabetes", start_offset=0, end_offset=32, section_title="Indications"),
        Chunk(chunk_id="c2", document_id="doc2", text="Contraindicated in severe renal impairment", start_offset=0, end_offset=42, section_title="Warnings")
    ]
    
    vector_store.add_chunks(chunks)
    
    # Search for "Metformin treats type 2 diabetes"
    results = vector_retrieve("Metformin treats type 2 diabetes", top_k=1)
    
    assert len(results) == 1
    item = results[0]
    assert item.type == "chunk"
    assert "Metformin treats type 2 diabetes" in item.content
    assert "[Section: Indications]" in item.content
    assert item.source_ids == ["doc1"]
    assert item.confidence > 0.0

def test_configurable_top_k():
    # Add multiple chunks
    chunks = [
        Chunk(chunk_id="c1", document_id="d1", text="Query matching text one", start_offset=0, end_offset=10),
        Chunk(chunk_id="c2", document_id="d2", text="Query matching text two", start_offset=0, end_offset=10),
        Chunk(chunk_id="c3", document_id="d3", text="Query matching text three", start_offset=0, end_offset=10)
    ]
    vector_store.add_chunks(chunks)
    
    # top_k = 2 should return at most 2 results
    results_2 = vector_retrieve("Query matching text one", top_k=2)
    assert len(results_2) == 2
    
    # top_k = 5 should return all 3 results (since ntotal=3)
    results_5 = vector_retrieve("Query matching text one", top_k=5)
    assert len(results_5) == 3

def test_empty_search_graceful():
    # Search on empty vector index should return [] gracefully without crashing
    results = vector_retrieve("Any query", top_k=5)
    assert results == []
