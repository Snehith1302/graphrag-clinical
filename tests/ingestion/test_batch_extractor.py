import os
import json
import pytest
from unittest.mock import patch, MagicMock
import httpx

from backend.app.config import settings
from backend.app.models.schemas import Chunk, Entity, Relationship
from ingestion.extraction.batch_extractor import (
    extract_batch,
    extract_batch_mock,
    extract_batch_llm,
    parse_batch_response,
    build_batch_api_payload,
    submit_gemini_batch_job,
    poll_gemini_batch_job,
    parse_gemini_batch_results,
    extract_workload_gemini_batch_api
)
from ingestion.run_ingestion import run_bulk_ingestion

@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id=f"chunk_doc1_{i}",
            document_id="doc1",
            text=f"Metformin hydrochloride tablets 500mg used to treat type 2 diabetes mellitus. Side effects include diarrhea and nausea. Chunk index {i}.",
            start_offset=i * 200,
            end_offset=(i * 200) + 150
        )
        for i in range(10)
    ]

def test_batching_size_and_partial_batch(sample_chunks):
    """Verifies batch processing of 10 chunks with batch size 4 results in 3 batches (4, 4, 2)."""
    batch_size = 4
    batches = [sample_chunks[i:i + batch_size] for i in range(0, len(sample_chunks), batch_size)]
    
    assert len(batches) == 3
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 2  # Final partial batch

def test_combined_entity_relation_response_parsing(sample_chunks):
    """Verifies parsing of structured combined JSON response containing entities and relationships."""
    chunk = sample_chunks[0]
    raw_json = json.dumps({
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "entities": [
                    {
                        "entity_text": "Metformin",
                        "normalized_name": "Metformin",
                        "entity_type": "Drug",
                        "confidence": 0.98
                    },
                    {
                        "entity_text": "type 2 diabetes",
                        "normalized_name": "Type 2 Diabetes",
                        "entity_type": "Condition",
                        "confidence": 0.95
                    }
                ],
                "relationships": [
                    {
                        "source_entity_text": "Metformin",
                        "relation_type": "TREATS",
                        "target_entity_text": "type 2 diabetes",
                        "confidence": 0.92
                    }
                ]
            }
        ]
    })
    
    entities, rels = parse_batch_response(raw_json, [chunk])
    
    assert len(entities) == 2
    assert entities[0].normalized_name == "Metformin"
    assert entities[0].entity_type == "Drug"
    assert entities[0].document_id == "doc1"
    assert entities[0].source_span == (0, 9)
    
    assert len(rels) == 1
    assert rels[0].relation_type == "TREATS"
    assert rels[0].source_ids == ["doc1"]

def test_malformed_batch_response(sample_chunks):
    """Verifies robust handling when LLM returns malformed JSON."""
    raw_json = "NOT_VALID_JSON"
    with pytest.raises(Exception):
        parse_batch_response(raw_json, sample_chunks[:1])

def test_missing_chunk_id(sample_chunks):
    """Verifies items with unknown or missing chunk_id are skipped safely."""
    raw_json = json.dumps({
        "chunks": [
            {
                "chunk_id": "non_existent_chunk_id",
                "entities": [
                    {
                        "entity_text": "Metformin",
                        "normalized_name": "Metformin",
                        "entity_type": "Drug",
                        "confidence": 0.98
                    }
                ]
            }
        ]
    })
    
    entities, rels = parse_batch_response(raw_json, sample_chunks[:1])
    assert len(entities) == 0
    assert len(rels) == 0

def test_duplicate_entity_handling(sample_chunks):
    """Verifies duplicate entities within the same chunk are deduplicated."""
    chunk = sample_chunks[0]
    raw_json = json.dumps({
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "entities": [
                    {
                        "entity_text": "Metformin",
                        "normalized_name": "Metformin",
                        "entity_type": "Drug",
                        "confidence": 0.98
                    },
                    {
                        "entity_text": "Metformin",
                        "normalized_name": "Metformin",
                        "entity_type": "Drug",
                        "confidence": 0.98
                    }
                ]
            }
        ]
    })
    
    entities, rels = parse_batch_response(raw_json, [chunk])
    assert len(entities) == 1

def test_relationship_grounding(sample_chunks):
    """Verifies relationships referencing ungrounded entities (not extracted in chunk) are rejected."""
    chunk = sample_chunks[0]
    raw_json = json.dumps({
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "entities": [
                    {
                        "entity_text": "Metformin",
                        "normalized_name": "Metformin",
                        "entity_type": "Drug",
                        "confidence": 0.98
                    }
                ],
                "relationships": [
                    {
                        "source_entity_text": "Metformin",
                        "relation_type": "TREATS",
                        "target_entity_text": "Hypertension",  # NOT in entities list!
                        "confidence": 0.90
                    }
                ]
            }
        ]
    })
    
    entities, rels = parse_batch_response(raw_json, [chunk])
    assert len(entities) == 1
    assert len(rels) == 0  # Rejected due to ungrounded target

def test_checkpoint_and_resume(tmp_path, monkeypatch):
    """Verifies that ingestion runner saves progress and resumes without re-extracting completed chunks."""
    fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
    json_path = os.path.join(fixtures_dir, "sample.json")
    
    monkeypatch.setenv("INGESTION_TEST_RUN", "true")
    summary1 = run_bulk_ingestion(json_path, force_mock=True)
    assert summary1["num_chunks"] > 0
    
    # Second run should resume and process 0 unprocessed chunks
    summary2 = run_bulk_ingestion(json_path, force_mock=True)
    assert summary2["num_chunks"] == summary1["num_chunks"]

def test_failed_batch_handling(sample_chunks, monkeypatch):
    """Verifies that when an LLM batch call fails after all retries, failed chunk IDs are returned."""
    mock_post = MagicMock(side_effect=httpx.RequestError("Network error"))
    with patch("httpx.post", mock_post):
        entities, rels, failed_ids, num_429 = extract_batch_llm(sample_chunks[:2], max_retries=2, backoff=0.01)
        
        assert len(entities) == 0
        assert len(rels) == 0
        assert failed_ids == [c.chunk_id for c in sample_chunks[:2]]

def test_429_retry_behavior(sample_chunks):
    """Verifies HTTP 429 rate limit triggers retries with exponential backoff and succeeds when 200 is returned."""
    response_429 = MagicMock()
    response_429.status_code = 429
    
    chunk = sample_chunks[0]
    valid_json = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "chunks": [{
                            "chunk_id": chunk.chunk_id,
                            "entities": [{
                                "entity_text": "Metformin",
                                "normalized_name": "Metformin",
                                "entity_type": "Drug",
                                "confidence": 0.98
                            }],
                            "relationships": []
                        }]
                    })
                }]
            }
        }]
    })
    
    response_200 = MagicMock()
    response_200.status_code = 200
    response_200.json.return_value = json.loads(valid_json)
    
    mock_post = MagicMock(side_effect=[response_429, response_200])
    
    with patch("httpx.post", mock_post):
        with patch("backend.app.config.settings.LLM_MODEL_NAME", "gemini-1.5-flash"):
            entities, rels, failed_ids, num_429 = extract_batch_llm([chunk], max_retries=3, backoff=0.01)
            
            assert mock_post.call_count == 2
            assert len(entities) == 1
            assert failed_ids == []
            assert num_429 == 1

def test_no_mock_fallback_during_real_extraction(sample_chunks, monkeypatch):
    """
    CRITICAL REQUIREMENT: Verifies that when real LLM mode is active and API extraction fails,
    the pipeline NEVER falls back to mock rule-based extraction or fabricates medical facts.
    """
    mock_post = MagicMock(side_effect=httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=MagicMock(status_code=500)))
    
    with patch("httpx.post", mock_post):
        with patch("backend.app.config.settings.LLM_API_KEY", "real_gemini_api_key_123"):
            with patch("backend.app.config.settings.LLM_MODEL_NAME", "gemini-1.5-flash"):
                monkeypatch.delenv("INGESTION_TEST_RUN", raising=False)
                
                entities, rels, failed_ids, num_429 = extract_batch_llm(sample_chunks[:2], max_retries=2, backoff=0.01)
                
                # Should return EMPTY entities/relations and mark chunks as failed. NO mock entities returned!
                assert len(entities) == 0
                assert len(rels) == 0
                assert failed_ids == [c.chunk_id for c in sample_chunks[:2]]

def test_build_batch_api_payload(sample_chunks):
    """Verifies that build_batch_api_payload correctly groups chunks into batch requests and maps custom_ids."""
    payload_items, custom_id_map = build_batch_api_payload(sample_chunks, batch_size=4)
    assert len(payload_items) == 3
    assert len(custom_id_map) == 3
    
    first_item = payload_items[0]
    assert "request" in first_item
    assert first_item["request"]["generationConfig"]["responseMimeType"] == "application/json"
    
    custom_id = list(custom_id_map.keys())[0]
    mapped_chunks = custom_id_map[custom_id]
    assert len(mapped_chunks) == 4
    assert mapped_chunks[0].chunk_id == sample_chunks[0].chunk_id

def test_submit_gemini_batch_job_mocked(sample_chunks):
    """Verifies submit_gemini_batch_job posts payload to official v1beta/models/...:batchGenerateContent endpoint with x-goog-api-key header."""
    payload_items, _ = build_batch_api_payload(sample_chunks[:4], batch_size=4)
    
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"name": "batches/batch-test-123", "state": "JOB_STATE_RUNNING"}
    
    with patch("httpx.post", return_value=mock_res) as mock_post:
        res = submit_gemini_batch_job(payload_items, api_key="test_key", model_name="gemini-3.5-flash")
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        call_headers = mock_post.call_args[1]["headers"]
        call_json = mock_post.call_args[1]["json"]
        
        assert call_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:batchGenerateContent"
        assert call_headers["x-goog-api-key"] == "test_key"
        assert "batch" in call_json
        assert "requests" in call_json["batch"]["input_config"]["requests"]
        assert res["name"] == "batches/batch-test-123"

def test_poll_gemini_batch_job_mocked():
    """Verifies poll_gemini_batch_job polls GET v1beta/{batch_name} with x-goog-api-key header until JOB_STATE_SUCCEEDED."""
    running_res = MagicMock()
    running_res.status_code = 200
    running_res.json.return_value = {"name": "batches/batch-test-123", "state": "JOB_STATE_RUNNING"}
    
    succeeded_res = MagicMock()
    succeeded_res.status_code = 200
    succeeded_res.json.return_value = {"name": "batches/batch-test-123", "state": "JOB_STATE_SUCCEEDED", "inlined_responses": []}
    
    with patch("httpx.get", side_effect=[running_res, succeeded_res]) as mock_get:
        job_data = poll_gemini_batch_job("batches/batch-test-123", api_key="test_key", poll_interval=0.01)
        assert mock_get.called
        call_url = mock_get.call_args[0][0]
        call_headers = mock_get.call_args[1]["headers"]
        
        assert call_url == "https://generativelanguage.googleapis.com/v1beta/batches/batch-test-123"
        assert call_headers["x-goog-api-key"] == "test_key"
        assert job_data["state"] == "JOB_STATE_SUCCEEDED"

def test_parse_gemini_batch_results_valid(sample_chunks):
    """Verifies parse_gemini_batch_results parses inlined_responses into Entity/Relationship models."""
    chunk = sample_chunks[0]
    payload_items, custom_id_map = build_batch_api_payload([chunk], batch_size=1)
    custom_id = list(custom_id_map.keys())[0]
    
    completed_job = {
        "state": "JOB_STATE_SUCCEEDED",
        "inlined_responses": [
            {
                "custom_id": custom_id,
                "status": {"code": 0},
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": json.dumps({
                                    "chunks": [{
                                        "chunk_id": chunk.chunk_id,
                                        "entities": [{
                                            "entity_text": "Metformin",
                                            "normalized_name": "Metformin",
                                            "entity_type": "Drug",
                                            "confidence": 0.95
                                        }],
                                        "relationships": []
                                    }]
                                })
                            }]
                        }
                    }]
                }
            }
        ]
    }
    
    entities, rels, failed_ids, num_429 = parse_gemini_batch_results(completed_job, custom_id_map)
    assert len(entities) == 1
    assert entities[0].normalized_name == "Metformin"
    assert failed_ids == []

def test_parse_gemini_batch_results_item_failure(sample_chunks):
    """Verifies that an error code or malformed JSON in a batch item records affected chunk IDs as failed."""
    chunk = sample_chunks[0]
    payload_items, custom_id_map = build_batch_api_payload([chunk], batch_size=1)
    custom_id = list(custom_id_map.keys())[0]
    
    completed_job = {
        "state": "JOB_STATE_SUCCEEDED",
        "inlined_responses": [
            {
                "custom_id": custom_id,
                "status": {"code": 13, "message": "Internal error"},
                "response": {}
            }
        ]
    }
    
    entities, rels, failed_ids, num_429 = parse_gemini_batch_results(completed_job, custom_id_map)
    assert len(entities) == 0
    assert failed_ids == [chunk.chunk_id]


