"""
Unit tests for the GraphRAG retrieval pipeline.
Mocks the Neo4j session runner and vector store metadata lookup for speed and offline stability.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.app.retrieval.graph_retrieve import graph_retrieve, extract_query_entities, resolve_provenance_chunks
from backend.app.retrieval.vector_store import vector_store

@pytest.fixture
def mock_neo4j_driver():
    with patch("backend.app.retrieval.graph_retrieve.neo4j_conn") as mock_conn:
        mock_conn.verify_health.return_value = True
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_conn.get_driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        yield mock_session

def test_extract_query_entities():
    # Test that Metformin is identified as a Drug in a clinical question
    ents = extract_query_entities("What are the side effects of Metformin?")
    assert len(ents) == 1
    assert ents[0]["normalized_name"] == "Metformin"
    assert ents[0]["entity_type"] == "Drug"
    assert ents[0]["entity_id"] == "drug_metformin"
    
    # Test multiple entities
    ents_multi = extract_query_entities("Can Metformin be combined with Insulin for diabetes?")
    names = {e["normalized_name"] for e in ents_multi}
    assert "Metformin" in names
    assert "Insulin" in names
    assert "Type 2 Diabetes" in names or "Diabetes" in names  # depending on pattern overlaps

def test_resolve_provenance_chunks():
    # Force initialize first to set _initialized=True and prevent disk reloads
    vector_store.initialize()
    # Seed vector store metadata
    vector_store.metadata = [
        {"document_id": "doc1", "text": "Metformin treats diabetes", "section_title": "Indications"},
        {"document_id": "doc2", "text": "Asthenia is a side effect", "section_title": "Adverse Reactions"}
    ]
    
    resolved = resolve_provenance_chunks(["doc2"])
    assert len(resolved) == 1
    assert resolved[0]["text"] == "Asthenia is a side effect"
    assert resolved[0]["section_title"] == "Adverse Reactions"

def test_graph_retrieve_neo4j_offline():
    # If health check returns False, return status graph_unavailable
    with patch("backend.app.retrieval.graph_retrieve.neo4j_conn") as mock_conn:
        mock_conn.verify_health.return_value = False
        res = graph_retrieve("What is Metformin?")
        assert res["status"] == "graph_unavailable"
        assert res["evidence_items"] == []

def test_graph_retrieve_insufficient_evidence():
    # If no clinical entities are found, return status insufficient_evidence
    with patch("backend.app.retrieval.graph_retrieve.neo4j_conn") as mock_conn:
        mock_conn.verify_health.return_value = True
        # Ask a query with zero medical terms
        res = graph_retrieve("Tell me a weather report.")
        assert res["status"] == "insufficient_evidence"

def test_graph_retrieve_one_hop_traversal(mock_neo4j_driver):
    # Mock node check to return Drug Metformin
    mock_node_record = MagicMock()
    mock_node_record.__getitem__.side_effect = lambda key: {
        "n": {"normalized_name": "Metformin", "confidence": 0.95},
        "label": "Drug"
    }[key]
    
    # Mock relationship record for 1-hop
    mock_rel_record = MagicMock()
    mock_rel_record.__getitem__.side_effect = lambda key: {
        "src": {"normalized_name": "Metformin"},
        "tgt": {"normalized_name": "Type 2 Diabetes"},
        "relation_type": "TREATS",
        "r": {"relation_id": "r1", "confidence": 0.95, "source_ids": ["doc_metformin_label"]}
    }[key]
    
    # Set run call returns sequence
    mock_result_node = MagicMock()
    mock_result_node.single.return_value = mock_node_record
    
    mock_result_rel = MagicMock()
    mock_result_rel.__iter__.return_value = [mock_rel_record]
    
    mock_neo4j_driver.run.side_effect = [mock_result_node, mock_result_rel]
    
    # Seed metadata chunk lookup
    vector_store.initialize()
    vector_store.metadata = [
        {"document_id": "doc_metformin_label", "text": "Metformin is indicated for Type 2 Diabetes.", "section_title": "Indications"}
    ]
    
    res = graph_retrieve("What is the therapeutic usage of Metformin?", max_hops=1)
    
    assert res["status"] == "ok"
    assert len(res["matched_graph_nodes"]) == 1
    assert res["matched_graph_nodes"][0]["normalized_name"] == "Metformin"
    
    assert len(res["graph_paths"]) == 1
    path = res["graph_paths"][0]
    assert path["source"] == "Metformin"
    assert path["relationship"] == "TREATS"
    assert path["target"] == "Type 2 Diabetes"
    
    assert len(res["evidence_items"]) == 1
    item = res["evidence_items"][0]
    assert item.type == "chunk"
    assert "Metformin is indicated for Type 2 Diabetes" in item.content
    assert item.source_ids == ["doc_metformin_label"]
    assert item.confidence == 0.95

def test_extract_query_entities_audited():
    # Test individual audited vocabulary entity recognition
    entities_to_test = ["Naproxen", "Mezereum", "Cyclosporine", "Aspirin", "Gout", "Quinolones", "Ofloxacin"]
    for ent in entities_to_test:
        ents = extract_query_entities(f"Tell me about {ent}.")
        assert len(ents) >= 1
        assert any(e["normalized_name"].lower() == ent.lower() for e in ents)

    # Test multi-entity query
    ents_multi = extract_query_entities("How does Naproxen compare with Cyclosporine and Aspirin?")
    names = {e["normalized_name"].lower() for e in ents_multi}
    assert "naproxen" in names
    assert "cyclosporine" in names
    assert "aspirin" in names

def test_live_graph_retrieve_q6_q9():
    """Runs actual GraphRAG traversals against live Neo4j for audited queries q6-q9."""
    from backend.app.graph.connection import neo4j_conn
    if not neo4j_conn.verify_health():
        pytest.skip("Neo4j database is offline. Skipping live integration tests.")

    # q6
    res_q6 = graph_retrieve("Which conditions are treated by the drug that interacts with cyclosporine?")
    assert res_q6["status"] == "ok"
    assert any(e["normalized_name"] == "Cyclosporine" for e in res_q6["matched_graph_nodes"])
    # should find path Naproxen - INTERACTS_WITH -> Cyclosporine
    assert len(res_q6["graph_paths"]) > 0
    assert any("INTERACTS_WITH" in p["relationship"] for p in res_q6["graph_paths"])

    # q7
    res_q7 = graph_retrieve("What is the relationship between the drug that treats gout and aspirin?")
    assert res_q7["status"] == "ok"
    assert any(e["normalized_name"] == "Aspirin" for e in res_q7["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Gout" for e in res_q7["matched_graph_nodes"])

    # q8
    res_q8 = graph_retrieve("Describe the graph path linking gout, naproxen, cyclosporine, and quinolones.")
    assert res_q8["status"] == "ok"
    assert any(e["normalized_name"] == "Gout" for e in res_q8["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Naproxen" for e in res_q8["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Cyclosporine" for e in res_q8["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Quinolones" for e in res_q8["matched_graph_nodes"])

    # q9
    res_q9 = graph_retrieve("How are aspirin, naproxen, cyclosporine, and quinolones connected through drug interactions?")
    assert res_q9["status"] == "ok"
    assert any(e["normalized_name"] == "Aspirin" for e in res_q9["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Naproxen" for e in res_q9["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Cyclosporine" for e in res_q9["matched_graph_nodes"])
    assert any(e["normalized_name"] == "Quinolones" for e in res_q9["matched_graph_nodes"])

def test_live_graph_retrieve_q5():
    """Runs actual GraphRAG traversal for q5 (Mezereum treats Pruritus) and verifies 1-hop paths are retrieved when max_hops >= 2."""
    from backend.app.graph.connection import neo4j_conn
    if not neo4j_conn.verify_health():
        pytest.skip("Neo4j database is offline. Skipping live integration tests.")

    res_q5 = graph_retrieve("What conditions are treated by Mezereum?", max_hops=2)
    assert res_q5["status"] == "ok"
    assert any(e["normalized_name"] == "Mezereum" for e in res_q5["matched_graph_nodes"])
    
    # Assert 1-hop path Mezereum -[TREATS]-> Pruritus is retrieved
    assert len(res_q5["graph_paths"]) > 0
    found_q5_path = any(
        (p.get("source") == "Mezereum" and p.get("target") == "Pruritus" and p.get("relationship") == "TREATS") or
        (p.get("source") == "Pruritus" and p.get("target") == "Mezereum" and p.get("relationship") == "TREATS")
        for p in res_q5["graph_paths"]
    )
    assert found_q5_path is True

def test_graph_retrieve_max_hop_depth_behavior():
    """Verifies that max_hops=1 only retrieves 1-hop paths, while max_hops=2 retrieves both 1-hop and 2-hop paths."""
    from backend.app.graph.connection import neo4j_conn
    if not neo4j_conn.verify_health():
        pytest.skip("Neo4j database is offline. Skipping live integration tests.")

    # With max_hops = 1, we should have only 1-hop paths
    res_hops_1 = graph_retrieve("How does Cyclosporine interact with Naproxen?", max_hops=1)
    assert res_hops_1["status"] == "ok"
    for p in res_hops_1["graph_paths"]:
        assert " -> " not in p["relationship"]

    # With max_hops = 2, we should get 2-hop paths (which contain mid-nodes)
    res_hops_2 = graph_retrieve("How does Cyclosporine interact with Naproxen?", max_hops=2)
    assert res_hops_2["status"] == "ok"
    has_2_hop = any(" -> " in p["relationship"] for p in res_hops_2["graph_paths"])
    assert has_2_hop is True
