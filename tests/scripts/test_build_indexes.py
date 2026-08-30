import os
import json
import pytest
from unittest.mock import patch, MagicMock
from scripts.build_indexes import build_data_stores
from backend.app.models.schemas import Chunk, Entity, Relationship

@pytest.fixture
def temp_processed_dir(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text="Metformin treats type 2 diabetes.",
            start_offset=0,
            end_offset=33
        )
    ]
    with open(d / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f)

    entities = [
        Entity(
            entity_id="Drug_metformin",
            entity_text="metformin",
            normalized_name="Metformin",
            entity_type="Drug",
            confidence=0.95,
            document_id="d1",
            source_span=[0, 9]
        )
    ]
    with open(d / "entities.json", "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in entities], f)

    rels = [
        Relationship(
            relation_id="rel_1",
            source_entity_id="Drug_metformin",
            relation_type="TREATS",
            target_entity_id="Condition_type_2_diabetes",
            confidence=0.90,
            source_ids=["d1"]
        )
    ]
    with open(d / "relationships.json", "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in rels], f)

    return str(d)

def test_build_data_stores_faiss_and_neo4j_offline(temp_processed_dir):
    """Verifies indexing invocation when Neo4j is offline."""
    with patch("scripts.build_indexes.vector_store") as mock_vs, \
         patch("scripts.build_indexes.neo4j_conn") as mock_conn:
        
        mock_vs.index = MagicMock()
        mock_vs.index.ntotal = 0
        mock_vs.metadata = []
        mock_conn.verify_health.return_value = False
        
        summary = build_data_stores(processed_dir=temp_processed_dir)
        
        assert summary["chunks_indexed"] == 1
        assert summary["neo4j_connection_status"] == "offline"
        assert summary["entities_loaded"] == 0
        assert summary["relationships_loaded"] == 0
        assert len(summary["errors_warnings"]) == 1
        assert "Neo4j is offline" in summary["errors_warnings"][0]
        assert mock_vs.add_chunks.called

def test_idempotent_faiss_rebuild_behavior(temp_processed_dir):
    """Verifies that FAISS indexing skips adding duplicate embeddings if index already matches chunk count."""
    with patch("scripts.build_indexes.vector_store") as mock_vs, \
         patch("scripts.build_indexes.neo4j_conn") as mock_conn:
        
        mock_vs.index = MagicMock()
        mock_vs.index.ntotal = 1
        mock_vs.metadata = ["dummy_meta"]
        mock_conn.verify_health.return_value = False
        
        summary = build_data_stores(processed_dir=temp_processed_dir)
        
        assert summary["chunks_indexed"] == 1
        # add_chunks should NOT be called because ntotal == len(chunks) == 1
        assert not mock_vs.add_chunks.called

def test_neo4j_load_invocation_online(temp_processed_dir):
    """Verifies Neo4j constraint initialization and graph insertion when Neo4j is online."""
    with patch("scripts.build_indexes.vector_store") as mock_vs, \
         patch("scripts.build_indexes.neo4j_conn") as mock_conn, \
         patch("scripts.build_indexes.initialize_constraints") as mock_init_constraints, \
         patch("scripts.build_indexes.insert_graph_data", return_value=True) as mock_insert:
        
        mock_vs.index = MagicMock()
        mock_vs.index.ntotal = 0
        mock_vs.metadata = []
        mock_conn.verify_health.return_value = True
        
        mock_session = MagicMock()
        mock_session.run.side_effect = [
            [{"label": "Drug", "cnt": 1}],      # Node query result
            [{"rel_type": "TREATS", "cnt": 1}]  # Relationship query result
        ]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_conn.get_driver.return_value = mock_driver

        summary = build_data_stores(processed_dir=temp_processed_dir)

        assert summary["neo4j_connection_status"] == "online"
        assert summary["entities_loaded"] == 1
        assert summary["relationships_loaded"] == 1
        assert summary["neo4j_node_counts"] == {"Drug": 1}
        assert summary["neo4j_relationship_counts"] == {"TREATS": 1}
        assert mock_init_constraints.called
        assert mock_insert.called

def test_summary_generation_structure(temp_processed_dir):
    """Verifies that build_data_stores returns a complete summary dictionary."""
    with patch("scripts.build_indexes.vector_store"), \
         patch("scripts.build_indexes.neo4j_conn") as mock_conn:
        
        mock_conn.verify_health.return_value = False
        summary = build_data_stores(processed_dir=temp_processed_dir)

        required_keys = [
            "chunks_indexed", "faiss_index_path", "entities_loaded",
            "relationships_loaded", "neo4j_connection_status",
            "neo4j_node_counts", "neo4j_relationship_counts", "errors_warnings"
        ]
        for k in required_keys:
            assert k in summary
