import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Neo4j Database Configuration
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="changeme")

    # LLM API Configuration
    LLM_API_KEY: str = Field(default="mock_key")
    LLM_MODEL_NAME: str = Field(default="mock_model")
    LLM_BATCH_SIZE: int = Field(default=4)
    LLM_BATCH_DELAY_SECONDS: float = Field(default=4.5)
    LLM_MAX_RETRIES: int = Field(default=1)
    LLM_BACKOFF_SECONDS: float = Field(default=5.0)

    # Gemini Batch API Configuration
    GEMINI_USE_BATCH_API: bool = Field(default=True)
    GEMINI_BATCH_POLL_INTERVAL_SECONDS: float = Field(default=10.0)
    GEMINI_BATCH_TIMEOUT_SECONDS: float = Field(default=3600.0)

    # Embeddings Model Configuration
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = Field(default="faiss")
    VECTOR_STORE_PATH: str = Field(default="./data/vector_index")

    # Retrieval System Tuning
    MAX_HOP_DEPTH: int = Field(default=3)
    VECTOR_TOP_K: int = Field(default=5)
    RELATION_CONFIDENCE_THRESHOLD: float = Field(default=0.6)

    # Ports and App Configuration
    BACKEND_PORT: int = Field(default=8000)
    FRONTEND_PORT: int = Field(default=5173)
    LOG_LEVEL: str = Field(default="info")
    FRONTEND_ORIGIN: str = Field(default="")

    # Load configuration from environment variables or .env file
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
