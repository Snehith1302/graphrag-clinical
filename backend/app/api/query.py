from fastapi import APIRouter, HTTPException
from backend.app.models.schemas import QueryRequest, GeneratedAnswer
from backend.app.retrieval.hybrid_retrieve import hybrid_retrieve
from backend.app.retrieval.vector_retrieve import vector_retrieve
from backend.app.retrieval.graph_retrieve import graph_retrieve
from backend.app.generation.answer_generator import generate_answer
from backend.app.config import settings
import logging

logger = logging.getLogger("graphrag.api.query")

router = APIRouter()

@router.post("/query", response_model=GeneratedAnswer)
async def query_endpoint(request: QueryRequest):
    """
    Receives a question, retrieves relevant context using the specified mode,
    generates a grounded answer, and returns citations, evidence chunks, and graph paths.
    """
    mode = request.mode.lower()
    if mode not in ["vector", "graph", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'vector', 'graph', or 'hybrid'.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    evidence_items = []
    graph_paths = []
    status = "ok"

    try:
        # 1. Fetch relevant evidence based on retrieval mode with fallbacks
        if mode == "vector":
            try:
                evidence_items = vector_retrieve(question, top_k=settings.VECTOR_TOP_K)
            except Exception as e:
                logger.warning(f"Vector retrieve failed, attempting graph fallback: {str(e)}")
                # Try graph retrieval fallback
                g_res = graph_retrieve(question, max_hops=settings.MAX_HOP_DEPTH)
                if g_res.get("status") not in ["graph_unavailable", "insufficient_evidence"]:
                    evidence_items = g_res.get("evidence_items", [])
                    graph_paths = g_res.get("graph_paths", [])
                    status = "vector_unavailable_fallback"
                else:
                    status = "insufficient_evidence"
                    
        elif mode == "graph":
            g_res = graph_retrieve(question, max_hops=settings.MAX_HOP_DEPTH)
            if g_res.get("status") == "graph_unavailable":
                logger.warning("Graph database offline, falling back to Vector RAG")
                try:
                    evidence_items = vector_retrieve(question, top_k=settings.VECTOR_TOP_K)
                    status = "graph_unavailable_fallback"
                except Exception as e:
                    logger.error(f"Vector fallback failed: {str(e)}")
                    status = "insufficient_evidence"
            else:
                evidence_items = g_res.get("evidence_items", [])
                graph_paths = g_res.get("graph_paths", [])
                status = g_res.get("status", "ok")
                
        else:  # hybrid
            h_res = hybrid_retrieve(question, top_k=settings.VECTOR_TOP_K, max_hops=settings.MAX_HOP_DEPTH)
            evidence_items = h_res.get("evidence_items", [])
            graph_paths = h_res.get("graph_paths", [])
            status = h_res.get("status", "ok")

        # 2. Invoke the shared generation service
        answer = generate_answer(question, evidence_items, mode)
        
        # 3. Enrich output response schemas
        answer.evidence = evidence_items
        answer.graph_paths = graph_paths
        
        # Map statuses cleanly
        if status in ["insufficient_evidence", "graph_unavailable_fallback", "vector_unavailable_fallback"]:
            answer.status = status
            
        # Check if LLM returns insufficient evidence state
        if answer.confidence == "insufficient_evidence":
            answer.status = "insufficient_evidence"

        return answer

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        # Raise HTTP 503 if the LLM provider fails or is timed out/unreachable
        raise HTTPException(status_code=503, detail="LLM generation service currently unavailable.")
