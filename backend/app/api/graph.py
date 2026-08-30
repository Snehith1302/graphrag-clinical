from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from backend.app.config import settings
from backend.app.graph.connection import neo4j_conn
from backend.app.models.schemas import GraphNode, GraphEdge
from backend.app.retrieval.vector_store import vector_store
import logging

logger = logging.getLogger("graphrag.api.graph")

router = APIRouter()

def get_node_id(node_properties: Dict[str, Any]) -> str:
    """
    Utility to resolve the unique node ID from its property dictionary.
    """
    for k, v in node_properties.items():
        if k.endswith("_id"):
            return str(v)
    return str(node_properties.get("normalized_name", "unknown"))

@router.get("/graph/neighborhood")
async def get_neighborhood(
    entity: str = Query(..., description="Normalized name of the clinical entity"),
    hop_depth: int = Query(settings.MAX_HOP_DEPTH, ge=1, le=3, description="Configurable traversal hop depth")
):
    """
    Returns graph neighborhood (nodes and edges) for a requested clinical entity.
    """
    if not neo4j_conn.verify_health():
        raise HTTPException(status_code=503, detail="Neo4j graph database is currently offline.")

    entity_name = entity.strip()
    if not entity_name:
        raise HTTPException(status_code=400, detail="Entity query parameter cannot be empty.")

    nodes_map: Dict[str, GraphNode] = {}
    edges_map: Dict[str, GraphEdge] = {}

    driver = neo4j_conn.get_driver()
    try:
        with driver.session() as session:
            if hop_depth == 1:
                query = """
                MATCH (n) WHERE n.normalized_name = $entity_name
                MATCH (n)-[r]-(tgt)
                RETURN n, labels(n)[0] AS n_label, r, type(r) AS r_type, id(r) AS r_id, tgt, labels(tgt)[0] AS tgt_label
                """
                res = session.run(query, entity_name=entity_name)
                for rec in res:
                    n_props = dict(rec["n"])
                    n_label = rec["n_label"]
                    n_id = get_node_id(n_props)
                    
                    tgt_props = dict(rec["tgt"])
                    tgt_label = rec["tgt_label"]
                    tgt_id = get_node_id(tgt_props)
                    
                    nodes_map[n_id] = GraphNode(id=n_id, label=n_label, name=n_props.get("normalized_name", ""), properties=n_props)
                    nodes_map[tgt_id] = GraphNode(id=tgt_id, label=tgt_label, name=tgt_props.get("normalized_name", ""), properties=tgt_props)
                    
                    edge_id = str(rec["r_id"])
                    edges_map[edge_id] = GraphEdge(
                        id=edge_id,
                        source=n_id,
                        target=tgt_id,
                        type=rec["r_type"],
                        properties=dict(rec["r"])
                    )
            else:
                # 2-hop or greater traversal
                query = """
                MATCH (n) WHERE n.normalized_name = $entity_name
                MATCH (n)-[r1]-(mid)-[r2]-(tgt)
                RETURN n, labels(n)[0] AS n_label, 
                       r1, type(r1) AS r1_type, id(r1) AS r1_id,
                       mid, labels(mid)[0] AS mid_label, 
                       r2, type(r2) AS r2_type, id(r2) AS r2_id,
                       tgt, labels(tgt)[0] AS tgt_label
                """
                res = session.run(query, entity_name=entity_name)
                for rec in res:
                    n_props = dict(rec["n"])
                    n_label = rec["n_label"]
                    n_id = get_node_id(n_props)
                    
                    mid_props = dict(rec["mid"])
                    mid_label = rec["mid_label"]
                    mid_id = get_node_id(mid_props)
                    
                    tgt_props = dict(rec["tgt"])
                    tgt_label = rec["tgt_label"]
                    tgt_id = get_node_id(tgt_props)
                    
                    nodes_map[n_id] = GraphNode(id=n_id, label=n_label, name=n_props.get("normalized_name", ""), properties=n_props)
                    nodes_map[mid_id] = GraphNode(id=mid_id, label=mid_label, name=mid_props.get("normalized_name", ""), properties=mid_props)
                    nodes_map[tgt_id] = GraphNode(id=tgt_id, label=tgt_label, name=tgt_props.get("normalized_name", ""), properties=tgt_props)
                    
                    edge1_id = str(rec["r1_id"])
                    edges_map[edge1_id] = GraphEdge(
                        id=edge1_id,
                        source=n_id,
                        target=mid_id,
                        type=rec["r1_type"],
                        properties=dict(rec["r1"])
                    )
                    
                    edge2_id = str(rec["r2_id"])
                    edges_map[edge2_id] = GraphEdge(
                        id=edge2_id,
                        source=mid_id,
                        target=tgt_id,
                        type=rec["r2_type"],
                        properties=dict(rec["r2"])
                    )

        return {
            "nodes": list(nodes_map.values()),
            "edges": list(edges_map.values())
        }
    except Exception as e:
        logger.error(f"Error querying neighborhood: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@router.get("/graph/stats")
async def get_stats():
    """
    Returns statistics of the Neo4j Graph Database and document corpus counts.
    """
    if not neo4j_conn.verify_health():
        raise HTTPException(status_code=503, detail="Neo4j graph database is currently offline.")

    node_counts = {}
    relationship_counts = {}
    total_nodes = 0
    total_relationships = 0

    driver = neo4j_conn.get_driver()
    try:
        with driver.session() as session:
            # 1. Count nodes by label
            node_res = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
            for rec in node_res:
                label = rec["label"] or "Unknown"
                cnt = rec["cnt"]
                node_counts[label] = cnt
                total_nodes += cnt
                
            # 2. Count relationships by type
            rel_res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt")
            for rec in rel_res:
                rel_type = rec["rel_type"]
                cnt = rec["cnt"]
                relationship_counts[rel_type] = cnt
                total_relationships += cnt

        # 3. Document Count from local vector store metadata
        vector_store.initialize()
        doc_ids = set()
        if vector_store.metadata:
            for meta in vector_store.metadata:
                if "document_id" in meta:
                    doc_ids.add(meta["document_id"])
        total_docs = len(doc_ids)

        return {
            "node_counts": node_counts,
            "relationship_counts": relationship_counts,
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
            "total_documents": total_docs
        }
    except Exception as e:
        logger.error(f"Error gathering stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
