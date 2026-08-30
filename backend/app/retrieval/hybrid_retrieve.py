"""
Hybrid retrieval module combining vector-based and graph-based retrieval.
Fuses evidence sets using a configurable weighted confidence ranking strategy,
handles partial failures, and deduplicates chunk outputs.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
from backend.app.config import settings
from backend.app.models.schemas import EvidenceItem
from backend.app.retrieval.vector_retrieve import vector_retrieve
from backend.app.retrieval.graph_retrieve import graph_retrieve

logger = logging.getLogger("graphrag.retrieval.hybrid")

def hybrid_retrieve(
    query: str, 
    top_k: int = 5, 
    max_hops: int = 3, 
    w_vector: float = 0.5, 
    w_graph: float = 0.5
) -> Dict[str, Any]:
    """
    Fuses Vector RAG and GraphRAG outputs for the same query.
    Applies weighted confidence scoring, deduplicates chunks, and resolves partial failures.
    Returns:
        Dict[str, Any]: Structured hybrid retrieval result dictionary.
    """
    logger.info(f"Performing hybrid retrieval for query: '{query}' (top_k={top_k}, max_hops={max_hops})")
    
    # 1. Execute Vector Retrieval
    vector_evidence: List[EvidenceItem] = []
    vector_ok = True
    try:
        vector_evidence = vector_retrieve(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Vector search failed in hybrid pipeline: {str(e)}")
        vector_ok = False
        
    # 2. Execute Graph Retrieval
    graph_res: Dict[str, Any] = {}
    graph_ok = True
    try:
        graph_res = graph_retrieve(query, max_hops=max_hops)
        if graph_res.get("status") == "graph_unavailable":
            graph_ok = False
            graph_res = {}
    except Exception as e:
        logger.error(f"Graph traversal failed in hybrid pipeline: {str(e)}")
        graph_ok = False
        graph_res = {}

    graph_evidence: List[EvidenceItem] = graph_res.get("evidence_items", [])
    graph_paths: List[Dict[str, Any]] = graph_res.get("graph_paths", [])

    # 3. Determine pipeline health status
    if not vector_ok and not graph_ok:
        return {
            "evidence_items": [],
            "graph_paths": [],
            "source_ids": [],
            "retrieval_methods_by_item": {},
            "status": "insufficient_evidence"
        }
        
    status = "ok"
    if not vector_ok:
        status = "graph_only"
    elif not graph_ok:
        status = "vector_only"
    elif not vector_evidence and not graph_evidence:
        status = "insufficient_evidence"

    # 4. Fusion and Deduplication
    # We index unique items by their content text
    unique_items: Dict[str, Dict[str, Any]] = {}
    
    # Process Vector Evidence
    if vector_ok:
        for idx, item in enumerate(vector_evidence):
            content = item.content
            unique_items[content] = {
                "item": item,
                "vector_score": item.confidence,
                "graph_score": 0.0,
                "methods": ["vector"]
            }

    # Process Graph Evidence
    if graph_ok:
        for item in graph_evidence:
            content = item.content
            if content in unique_items:
                unique_items[content]["graph_score"] = item.confidence
                if "graph" not in unique_items[content]["methods"]:
                    unique_items[content]["methods"].append("graph")
                # Merge source_ids
                existing_item = unique_items[content]["item"]
                existing_item.source_ids = list(set(existing_item.source_ids + item.source_ids))
            else:
                unique_items[content] = {
                    "item": item,
                    "vector_score": 0.0,
                    "graph_score": item.confidence,
                    "methods": ["graph"]
                }

    # 5. Compute Fused Scores and Construct Final Evidence Items
    fused_evidence: List[EvidenceItem] = []
    retrieval_methods_by_item: Dict[str, List[str]] = {}
    collected_source_ids = set()

    for content, details in unique_items.items():
        v_score = details["vector_score"]
        g_score = details["graph_score"]
        item: EvidenceItem = details["item"]
        
        # Weighted sum confidence fusion
        fused_score = (w_vector * v_score) + (w_graph * g_score)
        item.confidence = float(max(0.0, min(1.0, fused_score)))
        
        fused_evidence.append(item)
        retrieval_methods_by_item[content] = details["methods"]
        for s_id in item.source_ids:
            collected_source_ids.add(s_id)

    # 6. Rank evidence deterministically (sort by fused confidence descending, then by content string)
    fused_evidence.sort(key=lambda x: (-x.confidence, x.content))

    return {
        "evidence_items": fused_evidence,
        "graph_paths": graph_paths,
        "source_ids": list(collected_source_ids),
        "retrieval_methods_by_item": retrieval_methods_by_item,
        "status": status
    }
