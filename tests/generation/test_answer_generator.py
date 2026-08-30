"""
Unit tests for the LLM Answer Generation Service.
Uses mocked LLM responses for deterministic offline test coverage.
"""
import pytest
import httpx
from unittest.mock import patch, MagicMock
from backend.app.models.schemas import EvidenceItem
from backend.app.generation.answer_generator import (
    generate_answer, 
    is_personalized_query,
    validate_and_parse_citations,
    SAFETY_FOOTER
)

def test_is_personalized_query():
    assert is_personalized_query("Can I take Metformin during pregnancy?") is True
    assert is_personalized_query("should i take Metformin if I have diarrhea?") is True
    assert is_personalized_query("What is Metformin's mechanism of action?") is False

def test_personalized_query_refusal():
    # Personalized medical questions must return safety refusal block
    res = generate_answer("Can I take Metformin while breastfeeding?", [], "hybrid")
    assert "cannot provide personalized medical advice" in res.answer_text
    assert res.confidence == "insufficient_evidence"
    assert res.citations == []

def test_insufficient_evidence():
    # Empty evidence must yield insufficient evidence text
    res = generate_answer("What are the side effects of Metformin?", [], "vector")
    assert "I do not have sufficient evidence in the corpus to answer this." in res.answer_text
    assert res.confidence == "insufficient_evidence"

def test_citation_validation_and_parsing():
    allowed_sources = ["doc1", "doc2"]
    
    # 1. Valid citations
    raw_answer = "Metformin is useful [doc1]. Diarrhea is reported [doc2]."
    sanitized, markers = validate_and_parse_citations(raw_answer, allowed_sources)
    assert "[1]" in sanitized
    assert "[2]" in sanitized
    assert len(markers) == 2
    assert markers[0].source_id == "doc1"
    assert markers[0].marker == 1
    
    # 2. Invalid citations must be stripped
    raw_bad = "Some fake claim [doc3]. Metformin works [doc1]."
    sanitized_bad, markers_bad = validate_and_parse_citations(raw_bad, allowed_sources)
    assert "[doc3]" not in sanitized_bad
    assert "[1]" in sanitized_bad  # doc1 becomes marker 1
    assert len(markers_bad) == 1
    assert markers_bad[0].source_id == "doc1"

def test_grounded_mock_generation():
    evidence = [
        EvidenceItem(type="chunk", content="Metformin hydrochloride tablets are oral antihyperglycemic drugs.", source_ids=["doc_fda"], confidence=0.95)
    ]
    res = generate_answer("What is Metformin?", evidence, "vector")
    assert res.confidence == "high"
    assert "antihyperglycemic" in res.answer_text
    # Citation should be mapped to marker 1
    assert "[1]" in res.answer_text
    assert len(res.citations) == 1
    assert res.citations[0].source_id == "doc_fda"
    assert SAFETY_FOOTER in res.answer_text

@patch("backend.app.generation.answer_generator.call_real_llm")
def test_real_llm_json_generation_success(mock_call):
    # Mocking settings to run in real LLM mode
    with patch("backend.app.generation.answer_generator.settings") as mock_settings:
        mock_settings.LLM_API_KEY = "real_api_key"
        mock_settings.LLM_MODEL_NAME = "gpt-4"
        
        evidence = [
            EvidenceItem(type="chunk", content="Severe renal impairment contraindication.", source_ids=["doc1"], confidence=0.9)
        ]
        
        # Mock LLM returning valid JSON
        mock_call.return_value = {
            "answer_text": "Metformin is contraindicated in renal impairment [doc1].",
            "confidence": "high"
        }
        
        res = generate_answer("Is Metformin contraindicated?", evidence, "graph")
        assert res.confidence == "high"
        assert "[1]" in res.answer_text
        assert res.citations[0].source_id == "doc1"

@patch("backend.app.generation.answer_generator.httpx.post")
def test_llm_timeout_handling(mock_post):
    with patch("backend.app.generation.answer_generator.settings") as mock_settings:
        mock_settings.LLM_API_KEY = "real_api_key"
        mock_settings.LLM_MODEL_NAME = "gpt-4"
        
        # Trigger timeout exception
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")
        
        evidence = [
            EvidenceItem(type="chunk", content="Sample text.", source_ids=["doc1"], confidence=0.9)
        ]
        
        res = generate_answer("Question", evidence, "hybrid")
        assert "timed out" in res.answer_text
        assert res.confidence == "insufficient_evidence"
        assert res.citations == []
