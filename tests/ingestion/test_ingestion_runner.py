import os
from ingestion.run_ingestion import run_bulk_ingestion

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "openfda")

def test_runner_on_sample_json():
    json_path = os.path.join(FIXTURES_DIR, "sample.json")
    # Set INGESTION_TEST_RUN env var to allow it to run using mock fallback
    os.environ["INGESTION_TEST_RUN"] = "true"
    
    summary = run_bulk_ingestion(json_path, force_mock=True)
    
    assert summary["num_source_documents"] == 20
    assert summary["num_successfully_parsed"] == 1
    assert summary["num_chunks"] > 0
    assert "data/processed/documents.json" in summary["output_files"]
    assert "data/processed/chunks.json" in summary["output_files"]
    assert "data/processed/entities.json" in summary["output_files"]
    assert "data/processed/relationships.json" in summary["output_files"]

def test_runner_on_real_data_json():
    real_path = os.path.join(REAL_DATA_DIR, "openfda_labels_20.json")
    if not os.path.exists(real_path):
        return  # Skip if real dataset is not present in local test run environment
        
    os.environ["INGESTION_TEST_RUN"] = "true"
    summary = run_bulk_ingestion(real_path, force_mock=True)
    
    assert summary["num_successfully_parsed"] == 20
    assert summary["num_chunks"] > 0
    assert summary["num_entities"] > 0
    assert summary["num_relationships"] >= 0
    
    # Check that processed output files exist
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    assert os.path.exists(os.path.join(processed_dir, "documents.json"))
    assert os.path.exists(os.path.join(processed_dir, "chunks.json"))
    assert os.path.exists(os.path.join(processed_dir, "entities.json"))
    assert os.path.exists(os.path.join(processed_dir, "relationships.json"))
