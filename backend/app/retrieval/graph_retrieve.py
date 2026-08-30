"""
Graph-based traversal retrieval module.
Extracts clinical entities, links them to graph nodes, traverses Neo4j,
and resolves provenance using vector store metadata chunks.
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from backend.app.config import settings
from backend.app.graph.connection import neo4j_conn
from backend.app.models.schemas import EvidenceItem
from backend.app.retrieval.vector_store import vector_store
from ingestion.extraction.entity_extractor import MOCK_CLINICAL_DICTIONARY, generate_entity_id

logger = logging.getLogger("graphrag.retrieval.graph")

# Parameterized safe Cypher traversal templates
CYPHER_TEMPLATES = {
    "one_hop": """
    MATCH (src)
    WHERE src.drug_id = $node_id OR src.condition_id = $node_id OR src.symptom_id = $node_id 
       OR src.side_effect_id = $node_id OR src.population_id = $node_id OR src.study_id = $node_id 
       OR src.guideline_id = $node_id
    MATCH (src)-[r]-(tgt)
    RETURN src, labels(src)[0] AS src_label, r, type(r) AS relation_type, tgt, labels(tgt)[0] AS tgt_label
    """,
    "two_hop": """
    MATCH (src)
    WHERE src.drug_id = $node_id OR src.condition_id = $node_id OR src.symptom_id = $node_id 
       OR src.side_effect_id = $node_id OR src.population_id = $node_id OR src.study_id = $node_id 
       OR src.guideline_id = $node_id
    MATCH (src)-[r1]-(mid)-[r2]-(tgt)
    WHERE tgt <> src
    RETURN src, labels(src)[0] AS src_label, r1, type(r1) AS r1_type, mid, labels(mid)[0] AS mid_label, r2, type(r2) AS r2_type, tgt, labels(tgt)[0] AS tgt_label
    """
}

import os
import json

def load_canonical_dictionary() -> List[Tuple[str, str, str]]:
    """Loads all unique entities from processed entities dataset to construct the query entity parser vocabulary."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    entities_path = os.path.join(base_dir, "data", "processed", "entities.json")
    
    dict_items = []
    seen = set()
    
    if os.path.exists(entities_path):
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                ents = json.load(f)
            for e in ents:
                name = e["normalized_name"]
                etype = e["entity_type"]
                if (name.lower(), etype) not in seen:
                    seen.add((name.lower(), etype))
                    pattern = rf"(?i)\b{re.escape(name)}\b"
                    dict_items.append((pattern, etype, name))
        except Exception as err:
            logger.error(f"Error loading canonical entities for query parsing: {str(err)}")
            
    # Include fallback MOCK_CLINICAL_DICTIONARY items if they are not already present
    from ingestion.extraction.entity_extractor import MOCK_CLINICAL_DICTIONARY
    for pattern, etype, name in MOCK_CLINICAL_DICTIONARY:
        if (name.lower(), etype) not in seen:
            seen.add((name.lower(), etype))
            dict_items.append((pattern, etype, name))
            
    # Sort by canonical name length descending to match longer multi-word phrases first
    dict_items.sort(key=lambda x: len(x[2]), reverse=True)
    return dict_items

# Global active dictionary populated at runtime
ACTIVE_CLINICAL_DICTIONARY = load_canonical_dictionary()

def extract_query_entities(query_text: str) -> List[Dict[str, str]]:
    """
    Identifies clinical entities in the query text using the project's dictionary matches.
    """
    found = []
    global ACTIVE_CLINICAL_DICTIONARY
    # Reload if empty or if it only contains the fallback items but entities.json is now available
    if len(ACTIVE_CLINICAL_DICTIONARY) <= 15:
        ACTIVE_CLINICAL_DICTIONARY = load_canonical_dictionary()
        
    for pattern, ent_type, canonical_name in ACTIVE_CLINICAL_DICTIONARY:
        for match in re.finditer(pattern, query_text):
            ent_id = generate_entity_id(ent_type, canonical_name)
            if not any(f["entity_id"] == ent_id for f in found):
                found.append({
                    "entity_id": ent_id,
                    "normalized_name": canonical_name,
                    "entity_type": ent_type
                })
    return found

def resolve_provenance_chunks(source_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Finds text chunks in the vector store matching the provided source document IDs.
    """
    # Force initialize vector store metadata if empty
    vector_store.initialize()
    if not vector_store.metadata:
        return []
        
    resolved = []
    source_set = set(source_ids)
    for meta in vector_store.metadata:
        if meta.get("document_id") in source_set:
            resolved.append(meta)
    return resolved

def graph_retrieve(query: str, max_hops: Optional[int] = None) -> Dict[str, Any]:
    """
    Performs GraphRAG retrieval:
    1. Extracts clinical entities from query.
    2. Matches entities against Neo4j.
    3. Runs parameterized 1-hop or 2-hop traversal.
    4. Resolves edge provenance back to text chunks.
    5. Returns structured GraphRAG retrieval results.
    """
    hops = max_hops if max_hops is not None else settings.MAX_HOP_DEPTH
    logger.info(f"Performing graph retrieval (max_hops={hops}) for query: {query}")
    
    result_template = {
        "matched_query_entities": [],
        "matched_graph_nodes": [],
        "graph_paths": [],
        "evidence_items": [],
        "source_ids": [],
        "traversal_depth": hops,
        "retrieval_method": "graph",
        "status": "ok"
    }

    # 1. Neo4j availability validation
    if not neo4j_conn.verify_health():
        logger.warning("Graph retrieval requested but Neo4j is offline.")
        result_template["status"] = "graph_unavailable"
        return result_template

    # 2. Extract clinical query entities
    query_entities = extract_query_entities(query)
    result_template["matched_query_entities"] = query_entities
    if not query_entities:
        logger.info("No query entities detected. Graph retrieval complete (insufficient evidence).")
        result_template["status"] = "insufficient_evidence"
        return result_template

    # 3. Match nodes and traverse the graph
    matched_nodes = []
    graph_paths = []
    collected_source_ids = set()
    raw_evidence_list = []
    
    driver = neo4j_conn.get_driver()
    with driver.session() as session:
        for ent in query_entities:
            node_id = ent["entity_id"]
            
            # Check if this node exists in Neo4j
            check_query = """
            MATCH (n)
            WHERE n.drug_id = $node_id OR n.condition_id = $node_id OR n.symptom_id = $node_id 
               OR n.side_effect_id = $node_id OR n.population_id = $node_id OR n.study_id = $node_id 
               OR n.guideline_id = $node_id
            RETURN n, labels(n)[0] AS label LIMIT 1
            """
            check_res = session.run(check_query, node_id=node_id)
            record = check_res.single()
            if not record:
                # Entity unmatched in graph database
                continue
                
            node_props = dict(record["n"])
            node_label = record["label"]
            matched_nodes.append({
                "entity_id": node_id,
                "normalized_name": node_props.get("normalized_name"),
                "entity_type": node_label
            })
            
            # Perform parameterized Cypher traversal based on depth
            if hops >= 1:
                cypher = CYPHER_TEMPLATES["one_hop"]
                traversal_res = session.run(cypher, node_id=node_id)
                for rec in traversal_res:
                    src_name = dict(rec["src"]).get("normalized_name")
                    tgt_name = dict(rec["tgt"]).get("normalized_name")
                    rel_type = rec["relation_type"]
                    rel_props = dict(rec["r"])
                    
                    # Deduplicate 1-hop path
                    exists = False
                    for existing in graph_paths:
                        if existing.get("source") == src_name and existing.get("target") == tgt_name and existing.get("relationship") == rel_type:
                            exists = True
                            break
                    if not exists:
                        graph_paths.append({
                            "source": src_name,
                            "relationship": rel_type,
                            "target": tgt_name,
                            "properties": rel_props
                        })
                    
                    # Accumulate source document IDs
                    src_ids = rel_props.get("source_ids", [])
                    for s_id in src_ids:
                        collected_source_ids.add(s_id)
                        raw_evidence_list.append((s_id, rel_props.get("confidence", 0.9)))
                        
            if hops >= 2:
                # For hops >= 2, also run two-hop Cypher template
                cypher = CYPHER_TEMPLATES["two_hop"]
                traversal_res = session.run(cypher, node_id=node_id)
                for rec in traversal_res:
                    src_name = dict(rec["src"]).get("normalized_name")
                    mid_name = dict(rec["mid"]).get("normalized_name")
                    tgt_name = dict(rec["tgt"]).get("normalized_name")
                    
                    rel1_type = rec["r1_type"]
                    rel2_type = rec["r2_type"]
                    rel1_props = dict(rec["r1"])
                    rel2_props = dict(rec["r2"])
                    
                    rel_desc = f"{rel1_type} -> {mid_name} -> {rel2_type}"
                    # Deduplicate 2-hop path
                    exists = False
                    for existing in graph_paths:
                        if existing.get("source") == src_name and existing.get("target") == tgt_name and existing.get("relationship") == rel_desc:
                            exists = True
                            break
                    if not exists:
                        graph_paths.append({
                            "source": src_name,
                            "relationship": rel_desc,
                            "target": tgt_name,
                            "properties": {**rel1_props, **rel2_props}
                        })
                    
                    # Merge document source references from both relationships
                    src_ids = list(set(rel1_props.get("source_ids", []) + rel2_props.get("source_ids", [])))
                    avg_conf = (rel1_props.get("confidence", 0.9) + rel2_props.get("confidence", 0.9)) / 2.0
                    for s_id in src_ids:
                        collected_source_ids.add(s_id)
                        raw_evidence_list.append((s_id, avg_conf))

    result_template["matched_graph_nodes"] = matched_nodes
    result_template["graph_paths"] = graph_paths
    result_template["source_ids"] = list(collected_source_ids)
    
    if not matched_nodes or not graph_paths:
        logger.info("Graph traversal found no paths. Graph retrieval complete (insufficient evidence).")
        result_template["status"] = "insufficient_evidence"
        return result_template

    # 4. Resolve source document IDs to text chunks and deduplicate
    unique_evidence_items: Dict[str, EvidenceItem] = {}
    for s_id, conf in raw_evidence_list:
        chunks = resolve_provenance_chunks([s_id])
        for chk in chunks:
            sec_prefix = f"[Section: {chk['section_title']}] " if chk.get("section_title") else ""
            content = f"{sec_prefix}{chk['text']}"
            
            # Deduplicate by content to prevent overlap of identical paragraphs
            if content in unique_evidence_items:
                existing = unique_evidence_items[content]
                existing.confidence = max(existing.confidence, conf)
                if s_id not in existing.source_ids:
                    existing.source_ids.append(s_id)
            else:
                unique_evidence_items[content] = EvidenceItem(
                    type="chunk",
                    content=content,
                    source_ids=[s_id],
                    confidence=conf
                )
                
    # 5. Rank evidence deterministically (sort by confidence descending, then by content string)
    ranked_evidence = list(unique_evidence_items.values())
    ranked_evidence.sort(key=lambda x: (-x.confidence, x.content))
    result_template["evidence_items"] = ranked_evidence
    
    return result_template
