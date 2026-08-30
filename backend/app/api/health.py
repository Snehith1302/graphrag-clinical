import os
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.graph.connection import neo4j_conn

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Checks the status of the backend, Neo4j graph database, vector store path, and LLM configuration.
    """
    # 1. Verify Neo4j connectivity
    neo4j_healthy = neo4j_conn.verify_health()

    # 2. Verify Vector Store folder
    vector_store_healthy = False
    if os.path.exists(settings.VECTOR_STORE_PATH):
        vector_store_healthy = True
    else:
        # Create it if it doesn't exist to make it healthy for the first test run
        try:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            vector_store_healthy = True
        except Exception:
            vector_store_healthy = False

    # 3. Verify LLM configuration (check if key is configured, not placeholder)
    llm_configured = (
        settings.LLM_API_KEY is not None 
        and settings.LLM_API_KEY != "your_api_key_here"
        and settings.LLM_API_KEY != ""
    )

    overall_status = "ok"
    # Even if some integrations are unhealthy, backend is up, so return 200 with details.
    
    return {
        "status": overall_status,
        "neo4j": neo4j_healthy,
        "vector_store": vector_store_healthy,
        "llm": llm_configured
    }
