"""
Data Store Initialization Script.
Loads processed dataset files (chunks.json, entities.json, relationships.json)
and populates the FAISS vector store and Neo4j graph database idempotently.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List

from backend.app.config import settings
from backend.app.models.schemas import Chunk, Entity, Relationship
from backend.app.retrieval.vector_store import vector_store
from backend.app.graph.schema import initialize_constraints
from backend.app.graph.insert import insert_graph_data
from backend.app.graph.connection import neo4j_conn

logger = logging.getLogger("graphrag.scripts.build_indexes")

def build_data_stores(
    processed_dir: str = "data/processed",
    clear_existing_faiss: bool = False
) -> Dict[str, Any]:
    """
    Builds/populates the FAISS vector index and Neo4j graph database from processed dataset JSON files.
    
    Returns a summary dictionary containing:
    - chunks_indexed: int
    - faiss_index_path: str
    - entities_loaded: int
    - relationships_loaded: int
    - neo4j_connection_status: str ("online" / "offline")
    - neo4j_node_counts: Optional[Dict[str, int]]
    - neo4j_relationship_counts: Optional[Dict[str, int]]
    - errors_warnings: List[str]
    """
    summary: Dict[str, Any] = {
        "chunks_indexed": 0,
        "faiss_index_path": os.path.join(settings.VECTOR_STORE_PATH, "index.faiss"),
        "entities_loaded": 0,
        "relationships_loaded": 0,
        "neo4j_connection_status": "offline",
        "neo4j_node_counts": None,
        "neo4j_relationship_counts": None,
        "errors_warnings": []
    }

    chunks_file = os.path.join(processed_dir, "chunks.json")
    entities_file = os.path.join(processed_dir, "entities.json")
    relationships_file = os.path.join(processed_dir, "relationships.json")

    # 1. FAISS Vector Store Indexing
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            chunks = [Chunk(**c) for c in raw_chunks]
            
            if clear_existing_faiss:
                vector_store.clear()

            # Avoid duplicating embeddings if index is already populated idempotently
            vector_store.initialize()
            if vector_store.index and vector_store.index.ntotal == len(chunks) and len(vector_store.metadata) == len(chunks):
                logger.info(f"FAISS index already contains all {len(chunks)} chunks. Skipping duplicate embedding.")
                summary["chunks_indexed"] = len(chunks)
            else:
                vector_store.clear()
                vector_store.add_chunks(chunks)
                summary["chunks_indexed"] = len(chunks)
                logger.info(f"Successfully indexed {len(chunks)} chunks into FAISS vector store.")
        except Exception as e:
            err_msg = f"FAISS indexing failed: {str(e)}"
            logger.error(err_msg)
            summary["errors_warnings"].append(err_msg)
    else:
        err_msg = f"Missing chunks file: {chunks_file}"
        logger.warning(err_msg)
        summary["errors_warnings"].append(err_msg)

    # 2. Neo4j Graph Database Population
    entities: List[Entity] = []
    relationships: List[Relationship] = []

    if os.path.exists(entities_file):
        try:
            with open(entities_file, "r", encoding="utf-8") as f:
                raw_ents = json.load(f)
            entities = [Entity(**e) for e in raw_ents]
        except Exception as e:
            err_msg = f"Failed loading entities from {entities_file}: {str(e)}"
            logger.error(err_msg)
            summary["errors_warnings"].append(err_msg)

    if os.path.exists(relationships_file):
        try:
            with open(relationships_file, "r", encoding="utf-8") as f:
                raw_rels = json.load(f)
            relationships = [Relationship(**r) for r in raw_rels]
        except Exception as e:
            err_msg = f"Failed loading relationships from {relationships_file}: {str(e)}"
            logger.error(err_msg)
            summary["errors_warnings"].append(err_msg)

    if neo4j_conn.verify_health():
        summary["neo4j_connection_status"] = "online"
        try:
            # Initialize Neo4j schema constraints & indexes first
            initialize_constraints()

            # Insert nodes & relationships idempotently using MERGE
            success = insert_graph_data(entities, relationships)
            if success:
                summary["entities_loaded"] = len(entities)
                summary["relationships_loaded"] = len(relationships)
                
                # Fetch node & relationship counts from live database
                driver = neo4j_conn.get_driver()
                with driver.session() as session:
                    n_res = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
                    summary["neo4j_node_counts"] = {r["label"]: r["cnt"] for r in n_res if r["label"]}
                    
                    r_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt")
                    summary["neo4j_relationship_counts"] = {r["rel_type"]: r["cnt"] for r in r_res if r["rel_type"]}
            else:
                err_msg = "Neo4j graph insertion returned false."
                summary["errors_warnings"].append(err_msg)
        except Exception as e:
            err_msg = f"Neo4j population error: {str(e)}"
            logger.error(err_msg)
            summary["errors_warnings"].append(err_msg)
    else:
        summary["neo4j_connection_status"] = "offline"
        warn_msg = "Neo4j is offline or unavailable. Graph population skipped. (FAISS index creation completed)."
        logger.warning(warn_msg)
        summary["errors_warnings"].append(warn_msg)

    return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = build_data_stores()
    print("\n========================================")
    print("      DATA STORE BUILD SUMMARY")
    print("========================================")
    print(f"Chunks Indexed: {summary['chunks_indexed']}")
    print(f"FAISS Index Path: {summary['faiss_index_path']}")
    print(f"Entities Loaded: {summary['entities_loaded']}")
    print(f"Relationships Loaded: {summary['relationships_loaded']}")
    print(f"Neo4j Status: {summary['neo4j_connection_status']}")
    print(f"Neo4j Node Counts: {summary['neo4j_node_counts']}")
    print(f"Neo4j Relationship Counts: {summary['neo4j_relationship_counts']}")
    print(f"Errors/Warnings: {summary['errors_warnings']}")
    print("========================================\n")
