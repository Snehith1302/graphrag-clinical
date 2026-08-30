"""
Neo4j Graph Insertion and Query Helpers module.
Provides functions to insert validated entities/relationships and query the graph.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
from neo4j import Transaction, Session
from backend.app.graph.connection import neo4j_conn
from backend.app.models.schemas import Entity, Relationship
from ingestion.extraction.entity_extractor import ALLOWED_ENTITY_TYPES
from ingestion.extraction.relation_extractor import ALLOWED_RELATION_TYPES

logger = logging.getLogger("graphrag.graph.insert")

# Map of entity labels to their unique constraint property name as defined in schema.py
ENTITY_ID_PROPERTY_MAP = {
    "Drug": "drug_id",
    "Condition": "condition_id",
    "Symptom": "symptom_id",
    "SideEffect": "side_effect_id",
    "Population": "population_id",
    "ClinicalStudy": "study_id",
    "Guideline": "guideline_id"
}

def get_id_property_name(label: str) -> str:
    """Returns the property name representing the unique identifier for a node type."""
    if label not in ENTITY_ID_PROPERTY_MAP:
        raise ValueError(f"Unsupported node type/label: {label}")
    return ENTITY_ID_PROPERTY_MAP[label]

# ========================================================
# 1. Graph Insertion Cypher Helpers
# ========================================================

def insert_entity_tx(tx: Transaction, entity: Entity) -> None:
    """
    Inserts a single entity node into Neo4j using Cypher MERGE.
    Updates confidence if the incoming confidence is higher.
    """
    label = entity.entity_type
    if label not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"Entity type '{label}' is not allowed in Neo4j schema.")
        
    id_prop = get_id_property_name(label)
    
    # Parameterized Cypher query with strict label checks
    query = f"""
    MERGE (n:{label} {{{id_prop}: $entity_id}})
    ON CREATE SET n.normalized_name = $normalized_name,
                  n.entity_type = $entity_type,
                  n.confidence = $confidence
    ON MATCH SET n.confidence = case when $confidence > n.confidence then $confidence else n.confidence end
    """
    tx.run(
        query, 
        entity_id=entity.entity_id, 
        normalized_name=entity.normalized_name, 
        entity_type=label, 
        confidence=entity.confidence
    )

def insert_relationship_tx(tx: Transaction, relationship: Relationship, entity_type_map: Dict[str, str]) -> None:
    """
    Inserts a relationship edge between two nodes.
    - Resolves node labels from the entity_type_map.
    - Merges provenance references (source_ids) in a deduplicated list using standard Cypher.
    - Updates confidence if incoming confidence is higher.
    """
    rel_type = relationship.relation_type
    if rel_type not in ALLOWED_RELATION_TYPES:
        raise ValueError(f"Relationship type '{rel_type}' is not allowed in Neo4j schema.")
        
    src_id = relationship.source_entity_id
    tgt_id = relationship.target_entity_id
    
    src_label = entity_type_map.get(src_id)
    tgt_label = entity_type_map.get(tgt_id)
    
    if not src_label or not tgt_label:
        raise ValueError(f"Grounding missing: source '{src_id}' or target '{tgt_id}' node label not found.")
        
    src_id_prop = get_id_property_name(src_label)
    tgt_id_prop = get_id_property_name(tgt_label)
    
    # Pure Cypher list deduplication query (does not require external APOC plugins)
    query = f"""
    MATCH (src:{src_label} {{{src_id_prop}: $source_id}})
    MATCH (tgt:{tgt_label} {{{tgt_id_prop}: $target_id}})
    MERGE (src)-[r:{rel_type}]->(tgt)
    ON CREATE SET r.relation_id = $relation_id,
                  r.confidence = $confidence,
                  r.source_ids = $source_ids
    ON MATCH SET r.confidence = case when $confidence > r.confidence then $confidence else r.confidence end,
                 r.source_ids = r.source_ids + [x IN $source_ids WHERE NOT x IN r.source_ids]
    """
    tx.run(
        query,
        source_id=src_id,
        target_id=tgt_id,
        relation_id=relationship.relation_id,
        confidence=relationship.confidence,
        source_ids=relationship.source_ids
    )

def insert_graph_data(entities: List[Entity], relationships: List[Relationship]) -> bool:
    """
    Populates the Neo4j graph with validated entity nodes and relationship edges.
    Runs inside a single transactional block to guarantee database integrity.
    """
    if not neo4j_conn.verify_health():
        logger.warning("Neo4j connection is offline. Skipping graph database insertion.")
        return False
        
    try:
        driver = neo4j_conn.get_driver()
        entity_type_map = {e.entity_id: e.entity_type for e in entities}
        
        with driver.session() as session:
            session.execute_write(lambda tx: [insert_entity_tx(tx, ent) for ent in entities])
            session.execute_write(lambda tx: [insert_relationship_tx(tx, rel, entity_type_map) for rel in relationships])
            
        logger.info(f"Successfully inserted {len(entities)} nodes and {len(relationships)} edges into Neo4j.")
        return True
    except Exception as e:
        logger.error(f"Failed to populate graph database: {str(e)}")
        return False

# ========================================================
# 2. Graph Retrieval & Query Helper Methods
# ========================================================

def get_node_by_id(node_id: str, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Looks up a node's properties.
    If label is provided, executes an efficient query. Otherwise, scans matches across types.
    """
    if not neo4j_conn.verify_health():
        return None
        
    driver = neo4j_conn.get_driver()
    with driver.session() as session:
        if label:
            id_prop = get_id_property_name(label)
            query = f"MATCH (n:{label} {{{id_prop}: $node_id}}) RETURN n, labels(n)[0] AS label"
        else:
            query = """
            MATCH (n)
            WHERE n.drug_id = $node_id OR n.condition_id = $node_id OR n.symptom_id = $node_id 
               OR n.side_effect_id = $node_id OR n.population_id = $node_id OR n.study_id = $node_id 
               OR n.guideline_id = $node_id
            RETURN n, labels(n)[0] AS label LIMIT 1
            """
        result = session.run(query, node_id=node_id)
        record = result.single()
        if record:
            node = record["n"]
            label_name = record["label"]
            props = dict(node)
            props["label"] = label_name
            return props
    return None

def get_one_hop_neighborhood(node_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all direct neighbors of a node, including relationship types and direction.
    """
    if not neo4j_conn.verify_health():
        return []
        
    query = """
    MATCH (n)
    WHERE n.drug_id = $node_id OR n.condition_id = $node_id OR n.symptom_id = $node_id 
       OR n.side_effect_id = $node_id OR n.population_id = $node_id OR n.study_id = $node_id 
       OR n.guideline_id = $node_id
    MATCH (n)-[r]-(neighbor)
    RETURN labels(n)[0] AS source_label, n AS source,
           type(r) AS relation_type, r AS relationship,
           labels(neighbor)[0] AS neighbor_label, neighbor AS neighbor,
           startNode(r) = n AS is_outgoing
    """
    neighbors = []
    driver = neo4j_conn.get_driver()
    with driver.session() as session:
        result = session.run(query, node_id=node_id)
        for record in result:
            neighbors.append({
                "source_label": record["source_label"],
                "source": dict(record["source"]),
                "relation_type": record["relation_type"],
                "relationship": dict(record["relationship"]),
                "neighbor_label": record["neighbor_label"],
                "neighbor": dict(record["neighbor"]),
                "is_outgoing": record["is_outgoing"]
            })
    return neighbors

def get_two_hop_traversal(node_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves paths of length 2 radiating from a node.
    """
    if not neo4j_conn.verify_health():
        return []
        
    query = """
    MATCH (n)
    WHERE n.drug_id = $node_id OR n.condition_id = $node_id OR n.symptom_id = $node_id 
       OR n.side_effect_id = $node_id OR n.population_id = $node_id OR n.study_id = $node_id 
       OR n.guideline_id = $node_id
    MATCH path = (n)-[r1]-(neighbor)-[r2]-(two_hop)
    WHERE two_hop <> n
    RETURN labels(n)[0] AS start_label, n AS start_node,
           type(r1) AS r1_type, neighbor AS neighbor_node, labels(neighbor)[0] AS neighbor_label,
           type(r2) AS r2_type, two_hop AS final_node, labels(two_hop)[0] AS final_label
    LIMIT 50
    """
    paths = []
    driver = neo4j_conn.get_driver()
    with driver.session() as session:
        result = session.run(query, node_id=node_id)
        for record in result:
            paths.append({
                "start": {"label": record["start_label"], "properties": dict(record["start_node"])},
                "r1": record["r1_type"],
                "neighbor": {"label": record["neighbor_label"], "properties": dict(record["neighbor_node"])},
                "r2": record["r2_type"],
                "final": {"label": record["final_label"], "properties": dict(record["final_node"])}
            })
    return paths

def lookup_provenance_sources(source_doc_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all relationships and connected nodes backed by a specific source document ID.
    """
    if not neo4j_conn.verify_health():
        return []
        
    query = """
    MATCH (src)-[r]->(tgt)
    WHERE $source_id IN r.source_ids
    RETURN src, labels(src)[0] AS src_label,
           r, type(r) AS relation_type,
           tgt, labels(tgt)[0] AS tgt_label
    """
    elements = []
    driver = neo4j_conn.get_driver()
    with driver.session() as session:
        result = session.run(query, source_id=source_doc_id)
        for record in result:
            elements.append({
                "source": {"label": record["src_label"], "properties": dict(record["src"])},
                "relationship": {"type": record["relation_type"], "properties": dict(record["r"])},
                "target": {"label": record["tgt_label"], "properties": dict(record["tgt"])}
            })
    return elements

def get_graph_statistics() -> Dict[str, Any]:
    """
    Compiles database summary statistics: node counts by label and relationships by type.
    """
    stats = {
        "nodes": {},
        "relationships": {},
        "total_nodes": 0,
        "total_relationships": 0
    }
    
    if not neo4j_conn.verify_health():
        return stats
        
    driver = neo4j_conn.get_driver()
    with driver.session() as session:
        # 1. Fetch node counts
        node_result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count")
        for record in node_result:
            lbl = record["label"] or "Unlabeled"
            cnt = record["count"]
            stats["nodes"][lbl] = cnt
            stats["total_nodes"] += cnt
            
        # 2. Fetch relationship counts
        rel_result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count")
        for record in rel_result:
            tp = record["type"]
            cnt = record["count"]
            stats["relationships"][tp] = cnt
            stats["total_relationships"] += cnt
            
    return stats
