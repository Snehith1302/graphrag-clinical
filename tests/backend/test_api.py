"""
API integration tests for FastAPI routing endpoints.
Mocks service-level backends for isolated HTTP boundary verification.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app
from backend.app.models.schemas import EvidenceItem, GeneratedAnswer

client = TestClient(app)

# ========================================================
# POST /api/query
# ========================================================

@patch("backend.app.api.query.vector_retrieve")
def test_api_query_vector(mock_vector):
    # Mock vector search results
    mock_vector.return_value = [
        EvidenceItem(type="chunk", content="Metformin blocks glucose production.", source_ids=["doc_fda"], confidence=0.9)
    ]
    
    response = client.post("/api/query", json={"question": "What does Metformin do?", "mode": "vector"})
    assert response.status_code == 200
    data = response.json()
    assert "antihyperglycemic" in data["answer_text"] or "antihyperglycemic" in data["answer_text"].lower() or "metformin" in data["answer_text"].lower()
    assert data["mode_used"] == "vector"
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["content"] == "Metformin blocks glucose production."
    assert "Disclaimer:" in data["answer_text"]

@patch("backend.app.api.query.graph_retrieve")
def test_api_query_graph(mock_graph):
    # Mock graph retrieval results
    mock_graph.return_value = {
        "status": "ok",
        "evidence_items": [
            EvidenceItem(type="chunk", content="Contraindicated in severe renal impairment.", source_ids=["doc_warn"], confidence=0.95)
        ],
        "graph_paths": [
            {"source": "Metformin", "relationship": "CONTRAINDICATED_FOR", "target": "Renal Impairment", "properties": {}}
        ],
        "source_ids": ["doc_warn"]
    }
    
    response = client.post("/api/query", json={"question": "What is the contraindication?", "mode": "graph"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode_used"] == "graph"
    assert len(data["evidence"]) == 1
    assert len(data["graph_paths"]) == 1
    assert data["graph_paths"][0]["relationship"] == "CONTRAINDICATED_FOR"

def test_api_query_invalid_mode():
    response = client.post("/api/query", json={"question": "What is Metformin?", "mode": "invalid_mode"})
    assert response.status_code == 400
    assert "Invalid mode" in response.json()["detail"]

def test_api_query_empty_question():
    response = client.post("/api/query", json={"question": "   ", "mode": "hybrid"})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()

@patch("backend.app.api.query.hybrid_retrieve")
def test_api_query_insufficient_evidence(mock_hybrid):
    mock_hybrid.return_value = {
        "status": "insufficient_evidence",
        "evidence_items": [],
        "graph_paths": [],
        "source_ids": []
    }
    
    response = client.post("/api/query", json={"question": "Some query without details", "mode": "hybrid"})
    assert response.status_code == 200
    data = response.json()
    assert "do not have sufficient evidence" in data["answer_text"]
    assert data["confidence"] == "insufficient_evidence"

# ========================================================
# GET /api/graph/neighborhood
# ========================================================

@patch("backend.app.api.graph.neo4j_conn")
def test_graph_neighborhood_offline(mock_conn):
    mock_conn.verify_health.return_value = False
    response = client.get("/api/graph/neighborhood?entity=Metformin")
    assert response.status_code == 503
    assert "offline" in response.json()["detail"].lower()

@patch("backend.app.api.graph.neo4j_conn")
def test_graph_neighborhood_success(mock_conn):
    mock_conn.verify_health.return_value = True
    
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_conn.get_driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    # Mock Cypher output records
    mock_record = MagicMock()
    mock_record.__getitem__.side_effect = lambda key: {
        "n": {"normalized_name": "Metformin", "drug_id": "drug_metformin"},
        "n_label": "Drug",
        "tgt": {"normalized_name": "Type 2 Diabetes", "condition_id": "condition_t2d"},
        "tgt_label": "Condition",
        "r_type": "TREATS",
        "r_id": 1234,
        "r": {"confidence": 0.95}
    }[key]
    
    mock_result = MagicMock()
    mock_result.__iter__.return_value = [mock_record]
    mock_session.run.return_value = mock_result
    
    response = client.get("/api/graph/neighborhood?entity=Metformin&hop_depth=1")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert data["nodes"][0]["name"] == "Metformin"
    assert data["nodes"][1]["name"] == "Type 2 Diabetes"

# ========================================================
# GET /api/graph/stats
# ========================================================

@patch("backend.app.api.graph.neo4j_conn")
@patch("backend.app.api.graph.vector_store")
def test_graph_stats(mock_store, mock_conn):
    mock_conn.verify_health.return_value = True
    mock_store.metadata = [{"document_id": "doc1"}, {"document_id": "doc2"}]
    
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_conn.get_driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    # Mock count results
    mock_rec_node = MagicMock()
    mock_rec_node.__getitem__.side_effect = lambda key: {"label": "Drug", "cnt": 5}[key]
    
    mock_rec_rel = MagicMock()
    mock_rec_rel.__getitem__.side_effect = lambda key: {"rel_type": "TREATS", "cnt": 3}[key]
    
    mock_res_node = MagicMock()
    mock_res_node.__iter__.return_value = [mock_rec_node]
    
    mock_res_rel = MagicMock()
    mock_res_rel.__iter__.return_value = [mock_rec_rel]
    
    mock_session.run.side_effect = [mock_res_node, mock_res_rel]
    
    response = client.get("/api/graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] == 5
    assert data["total_relationships"] == 3
    assert data["total_documents"] == 2

# ========================================================
# GET /api/evidence/{source_id}
# ========================================================

@patch("backend.app.api.evidence.vector_store")
def test_evidence_lookup(mock_store):
    mock_store.metadata = [
        {"document_id": "doc_metformin_label", "text": "Metformin treats diabetes.", "section_title": "Indications"}
    ]
    
    # Valid source ID lookup
    response = client.get("/api/evidence/doc_metformin_label")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc_metformin_label"
    assert "Metformin treats diabetes" in data["excerpt"]
    assert data["section"] == "Indications"
    
    # Invalid lookup returns HTTP 404
    response_fail = client.get("/api/evidence/doc_missing")
    assert response_fail.status_code == 404
    assert "not found" in response_fail.json()["detail"].lower()
