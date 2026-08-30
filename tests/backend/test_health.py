"""
Tests for the health-check API endpoint.
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    """Tests the root endpoint returns welcome info."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "health_url" in response.json()

def test_health_check_unauthenticated():
    """Tests `/api/health` returns status dict."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "neo4j" in data
    assert "vector_store" in data
    assert "llm" in data
