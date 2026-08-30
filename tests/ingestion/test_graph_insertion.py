"""
Unit tests for Neo4j Graph Population and Retrieval Helpers.
Uses unittest.mock to mock Neo4j connection driver, sessions, and transactions.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.app.models.schemas import Entity, Relationship
from backend.app.graph.insert import (
    insert_entity_tx,
    insert_relationship_tx,
    insert_graph_data,
    get_node_by_id,
    get_one_hop_neighborhood,
    get_two_hop_traversal,
    lookup_provenance_sources,
    get_graph_statistics
)

@pytest.fixture
def mock_transaction():
    tx = MagicMock()
    # Mock tx.run to return a mock result
    tx.run.return_value = MagicMock()
    return tx

def test_insert_all_node_types_cypher(mock_transaction):
    # Verify that all 7 allowed node types generate the correct parameterized cypher MERGE statement
    node_types_to_test = [
        ("Drug", "drug_id"),
        ("Condition", "condition_id"),
        ("Symptom", "symptom_id"),
        ("SideEffect", "side_effect_id"),
        ("Population", "population_id"),
        ("ClinicalStudy", "study_id"),
        ("Guideline", "guideline_id")
    ]
    
    for label, id_field in node_types_to_test:
        ent = Entity(
            entity_id=f"{label.lower()}_test",
            normalized_name="Test Name",
            entity_type=label,
            confidence=0.9,
            document_id="doc1",
            source_span=(0, 5)
        )
        
        insert_entity_tx(mock_transaction, ent)
        
        # Verify that tx.run was called
        args, kwargs = mock_transaction.run.call_args
        query_str = args[0]
        params = kwargs
        
        # Cypher must contain the label and unique ID property name
        assert f"MERGE (n:{label} {{{id_field}: $entity_id}})" in query_str
        assert "ON CREATE SET" in query_str
        assert "ON MATCH SET" in query_str
        assert params["entity_id"] == f"{label.lower()}_test"
        assert params["normalized_name"] == "Test Name"
        assert params["confidence"] == 0.9

def test_insert_relationship_cypher(mock_transaction):
    # Test relationship insert with TREATS Drug -> Condition
    rel = Relationship(
        relation_id="drug_test_treats_condition_test",
        source_entity_id="drug_test",
        relation_type="TREATS",
        target_entity_id="condition_test",
        confidence=0.85,
        source_ids=["doc1"]
    )
    
    entity_type_map = {
        "drug_test": "Drug",
        "condition_test": "Condition"
    }
    
    insert_relationship_tx(mock_transaction, rel, entity_type_map)
    
    args, kwargs = mock_transaction.run.call_args
    query_str = args[0]
    params = kwargs
    
    # Must run MATCH on Drug and Condition nodes, and MERGE the TREATS edge
    assert "MATCH (src:Drug {drug_id: $source_id})" in query_str
    assert "MATCH (tgt:Condition {condition_id: $target_id})" in query_str
    assert "MERGE (src)-[r:TREATS]->(tgt)" in query_str
    # Cypher list deduplication check
    assert "r.source_ids = r.source_ids + [x IN $source_ids WHERE NOT x IN r.source_ids]" in query_str
    
    assert params["source_id"] == "drug_test"
    assert params["target_id"] == "condition_test"
    assert params["confidence"] == 0.85
    assert params["source_ids"] == ["doc1"]

def test_invalid_label_or_relationship_type_rejection(mock_transaction):
    # 1. Invalid Label Node
    bad_ent = Entity(
        entity_id="doctor_alice",
        normalized_name="Alice",
        entity_type="Doctor",  # Unsupported
        confidence=0.9,
        document_id="doc1",
        source_span=(0, 5)
    )
    with pytest.raises(ValueError):
        insert_entity_tx(mock_transaction, bad_ent)
        
    # 2. Invalid Relationship Type
    bad_rel = Relationship(
        relation_id="drug_test_cures_condition_test",
        source_entity_id="drug_test",
        relation_type="CURES",  # Unsupported
        target_entity_id="condition_test",
        confidence=0.9,
        source_ids=["doc1"]
    )
    with pytest.raises(ValueError):
        insert_relationship_tx(mock_transaction, bad_rel, {"drug_test": "Drug", "condition_test": "Condition"})

@patch("backend.app.graph.insert.neo4j_conn")
def test_insert_graph_data_orchestrator(mock_conn):
    # Mock verify_health to be True
    mock_conn.verify_health.return_value = True
    
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_conn.get_driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 9))
    ]
    relationships = [
        Relationship(relation_id="rel_1", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.8, source_ids=["doc1"])
    ]
    
    # We map "condition_diabetes" to Condition in type map
    entity_type_map = {"drug_metformin": "Drug", "condition_diabetes": "Condition"}
    
    success = insert_graph_data(entities, relationships)
    assert success is True
    
    # Session must execute write transactions
    assert mock_session.execute_write.call_count == 2

@patch("backend.app.graph.insert.neo4j_conn")
def test_retrieval_helpers_offline_fallback(mock_conn):
    # If connection verification is offline, helpers should return clean fallbacks gracefully
    mock_conn.verify_health.return_value = False
    
    assert get_node_by_id("drug_metformin") is None
    assert get_one_hop_neighborhood("drug_metformin") == []
    assert get_two_hop_traversal("drug_metformin") == []
    assert lookup_provenance_sources("doc1") == []
    
    stats = get_graph_statistics()
    assert stats["total_nodes"] == 0
    assert stats["total_relationships"] == 0

@patch("backend.app.graph.insert.neo4j_conn")
def test_get_node_by_id(mock_conn):
    mock_conn.verify_health.return_value = True
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_conn.get_driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    # Mock records returned by session.run
    mock_record = MagicMock()
    mock_record.__getitem__.side_effect = lambda key: {
        "n": {"normalized_name": "Metformin", "confidence": 0.95},
        "label": "Drug"
    }[key]
    
    mock_result = MagicMock()
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result
    
    node = get_node_by_id("drug_metformin", label="Drug")
    assert node is not None
    assert node["normalized_name"] == "Metformin"
    assert node["label"] == "Drug"
    
    # Check session.run arguments
    args, kwargs = mock_session.run.call_args
    assert "MATCH (n:Drug {drug_id: $node_id})" in args[0]
    assert kwargs["node_id"] == "drug_metformin"
