import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api import health, query, graph, evidence
from backend.app.graph.schema import initialize_constraints
from backend.app.graph.connection import neo4j_conn

# Setup logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("graphrag.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle events.
    """
    logger.info("Initializing GraphRAG Clinical Decision Support system...")
    
    # Run Neo4j schema constraint initialization on startup
    initialize_constraints()
    
    yield
    
    # Close resources on shutdown
    logger.info("Shutting down GraphRAG Clinical Decision Support system...")
    neo4j_conn.close()

app = FastAPI(
    title="GraphRAG Clinical Knowledge Retrieval System API",
    description="A research prototype backend comparing graph-based and vector-based retrieval on public clinical literature.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
if settings.FRONTEND_ORIGIN:
    extra_origins = [o.strip() for o in settings.FRONTEND_ORIGIN.split(",") if o.strip()]
    origins.extend(extra_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(graph.router, prefix="/api", tags=["graph"])
app.include_router(evidence.router, prefix="/api", tags=["evidence"])

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the GraphRAG Clinical Retrieval API",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {settings.BACKEND_PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=settings.BACKEND_PORT, reload=True)
