from fastapi import APIRouter, HTTPException
from backend.app.retrieval.vector_store import vector_store
from backend.app.models.schemas import Evidence
import logging

logger = logging.getLogger("graphrag.api.evidence")

router = APIRouter()

@router.get("/evidence/{source_id}", response_model=Evidence)
async def get_evidence(source_id: str):
    """
    Returns citation/document metadata and the first evidence excerpt for a requested source_id.
    """
    vector_store.initialize()
    if not vector_store.metadata:
        raise HTTPException(status_code=404, detail="No evidence metadata registered in system.")

    clean_source_id = source_id.strip()
    
    # Search metadata ledger for matching chunks
    matching_chunk = None
    for chunk in vector_store.metadata:
        if chunk.get("document_id") == clean_source_id or chunk.get("chunk_id") == clean_source_id:
            matching_chunk = chunk
            break

    if not matching_chunk:
        raise HTTPException(status_code=404, detail=f"Evidence source '{clean_source_id}' not found in corpus.")

    # Format human-friendly title
    doc_id = matching_chunk.get("document_id", clean_source_id)
    doc_title = doc_id.replace("_", " ").replace("-", " ").title()

    return Evidence(
        source_id=clean_source_id,
        document_id=doc_id,
        title=doc_title,
        section=matching_chunk.get("section_title"),
        excerpt=matching_chunk.get("text", ""),
        url=f"https://clinical-repository.org/doc/{doc_id}"
    )
