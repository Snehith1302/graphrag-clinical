"""
Unit tests for the Hybrid Retrieval module.
Mocks the vector_retrieve and graph_retrieve endpoints for clean test boundary separation.
"""
import pytest
from unittest.mock import patch, MagicMock
from backend.app.models.schemas import EvidenceItem
from backend.app.retrieval.hybrid_retrieve import hybrid_retrieve

@pytest.fixture
def mock_retrieval_functions():
    with patch("backend.app.retrieval.hybrid_retrieve.vector_retrieve") as mock_vector, \
         patch("backend.app.retrieval.hybrid_retrieve.graph_retrieve") as mock_graph:
         
         yield mock_vector, mock_graph

def test_hybrid_retrieve_successful_fusion(mock_retrieval_functions):
    mock_vector, mock_graph = mock_retrieval_functions
    
    # Vector returns chunk A
    mock_vector.return_value = [
        EvidenceItem(type="chunk", content="Metformin treats diabetes", source_ids=["doc_vector"], confidence=0.8)
    ]
    
    # Graph returns chunk A (duplicate) and chunk B
    mock_graph.return_value = {
        "status": "ok",
        "graph_paths": [{"source": "Metformin", "relationship": "TREATS", "target": "Diabetes", "properties": {}}],
        "source_ids": ["doc_graph"],
        "evidence_items": [
            EvidenceItem(type="chunk", content="Metformin treats diabetes", source_ids=["doc_graph"], confidence=0.9),
            EvidenceItem(type="chunk", content="Severe renal impairment warning", source_ids=["doc_graph"], confidence=0.95)
        ]
    }
    
    res = hybrid_retrieve("Metformin diabetes query", top_k=5, max_hops=3, w_vector=0.5, w_graph=0.5)
    
    assert res["status"] == "ok"
    assert len(res["evidence_items"]) == 2  # Deduplicated from 3 items down to 2
    
    # Check that Chunk A has fused score: 0.5 * 0.8 + 0.5 * 0.9 = 0.85
    # Chunk B has fused score: 0.5 * 0.0 + 0.5 * 0.95 = 0.475
    # So Chunk A should be first (0.85 > 0.475)
    first_item = res["evidence_items"][0]
    assert first_item.content == "Metformin treats diabetes"
    assert first_item.confidence == pytest.approx(0.85)
    # Source IDs should be merged
    assert set(first_item.source_ids) == {"doc_vector", "doc_graph"}
    
    # Verify graph paths are preserved
    assert len(res["graph_paths"]) == 1
    assert res["graph_paths"][0]["source"] == "Metformin"

def test_vector_only_fallback(mock_retrieval_functions):
    mock_vector, mock_graph = mock_retrieval_functions
    
    mock_vector.return_value = [
        EvidenceItem(type="chunk", content="Metformin clinical text", source_ids=["doc1"], confidence=0.7)
    ]
    # Graph is offline
    mock_graph.return_value = {"status": "graph_unavailable"}
    
    res = hybrid_retrieve("Metformin", top_k=5, max_hops=3)
    assert res["status"] == "vector_only"
    assert len(res["evidence_items"]) == 1
    assert res["evidence_items"][0].content == "Metformin clinical text"

def test_graph_only_fallback(mock_retrieval_functions):
    mock_vector, mock_graph = mock_retrieval_functions
    
    # Vector is offline (returns exception or empty)
    mock_vector.side_effect = Exception("Vector index offline")
    
    # Graph is online
    mock_graph.return_value = {
        "status": "ok",
        "graph_paths": [],
        "source_ids": ["doc_graph"],
        "evidence_items": [
            EvidenceItem(type="chunk", content="Graph node clinical text", source_ids=["doc_graph"], confidence=0.9)
        ]
    }
    
    res = hybrid_retrieve("Diabetes", top_k=5, max_hops=3)
    assert res["status"] == "graph_only"
    assert len(res["evidence_items"]) == 1
    assert res["evidence_items"][0].content == "Graph node clinical text"

def test_both_unavailable(mock_retrieval_functions):
    mock_vector, mock_graph = mock_retrieval_functions
    
    mock_vector.side_effect = Exception("Offline")
    mock_graph.return_value = {"status": "graph_unavailable"}
    
    res = hybrid_retrieve("Any", top_k=5)
    assert res["status"] == "insufficient_evidence"
    assert res["evidence_items"] == []

def test_configurable_weights(mock_retrieval_functions):
    mock_vector, mock_graph = mock_retrieval_functions
    
    # Vector item (0.8) vs Graph item (0.9)
    mock_vector.return_value = [
        EvidenceItem(type="chunk", content="Vector text", source_ids=["d1"], confidence=0.8)
    ]
    mock_graph.return_value = {
        "status": "ok",
        "graph_paths": [],
        "source_ids": ["d2"],
        "evidence_items": [
            EvidenceItem(type="chunk", content="Graph text", source_ids=["d2"], confidence=0.9)
        ]
    }
    
    # 1. Vector heavy: w_vector = 0.9, w_graph = 0.1
    # Vector text score: 0.9 * 0.8 = 0.72
    # Graph text score: 0.1 * 0.9 = 0.09
    # So Vector text should rank first
    res_vec_heavy = hybrid_retrieve("Test", w_vector=0.9, w_graph=0.1)
    assert res_vec_heavy["evidence_items"][0].content == "Vector text"
    
    # 2. Graph heavy: w_vector = 0.1, w_graph = 0.9
    # Vector text score: 0.1 * 0.8 = 0.08
    # Graph text score: 0.9 * 0.9 = 0.81
    # So Graph text should rank first
    res_graph_heavy = hybrid_retrieve("Test", w_vector=0.1, w_graph=0.9)
    assert res_graph_heavy["evidence_items"][0].content == "Graph text"
