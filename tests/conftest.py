import os
import sys

# Force mock LLM configuration during unit testing
os.environ["LLM_API_KEY"] = "mock_key"
os.environ["LLM_MODEL_NAME"] = "mock_model"
os.environ["INGESTION_TEST_RUN"] = "true"
