"""
Vector-based semantic retrieval module.
"""
import logging
from typing import List
from backend.app.models.schemas import EvidenceItem
from backend.app.retrieval.vector_store import vector_store

logger = logging.getLogger("graphrag.retrieval.vector")

def vector_retrieve(query: str, top_k: int = 5) -> List[EvidenceItem]:
    """
    Retrieves the top_k relevant text chunks from the local FAISS vector database.
    """
    logger.info(f"Performing vector retrieval for query: {query} (top_k={top_k})")
    try:
        return vector_store.search(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Vector retrieval failed: {str(e)}")
        return []
