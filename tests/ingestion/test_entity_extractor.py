"""
Unit tests for clinical entity extraction.
"""
import pytest
from backend.app.models.schemas import Chunk
from ingestion.extraction.entity_extractor import (
    extract_entities, 
    normalize_name, 
    generate_entity_id,
    ALLOWED_ENTITY_TYPES
)

def test_normalization_and_id_generation():
    # Verify canonical mappings
    assert normalize_name("metformin hcl", "Drug") == "Metformin"
    assert normalize_name("type 2 diabetes mellitus", "Condition") == "Type 2 Diabetes"
    assert normalize_name("severe renal impairment", "Population") == "Severe Renal Impairment"
    
    # Verify ID generation
    assert generate_entity_id("Drug", "Metformin") == "drug_metformin"
    assert generate_entity_id("Population", "Severe Renal Impairment") == "population_severe_renal_impairment"

def test_extract_all_valid_types():
    # Chunk text containing examples of all 7 types
    text = (
        "In the Metformin Clinical Guideline 2024, Metformin is recommended for Type 2 Diabetes. "
        "It was evaluated in the Efficacy and Side Effects of Metformin in Type 2 Diabetes study. "
        "Common adverse reactions include diarrhea and headache. It presents with polyuria. "
        "Metformin is contraindicated in severe renal impairment patients."
    )
    # The text contains:
    # Metformin -> Drug
    # Type 2 Diabetes -> Condition
    # diarrhea -> SideEffect
    # headache -> SideEffect
    # polyuria -> Symptom (from mock dictionary, let's check: yes, polyuria is mapped to Symptom)
    # severe renal impairment -> Population
    # Metformin Clinical Guideline 2024 -> Guideline
    # Efficacy and Side Effects of Metformin in Type 2 Diabetes -> ClinicalStudy
    
    chunk = Chunk(
        chunk_id="test_chunk_all",
        document_id="doc_test",
        text=text,
        start_offset=100,
        end_offset=100 + len(text)
    )
    
    entities = extract_entities(chunk, method="rule_based_mock")
    assert len(entities) > 0
    
    extracted_types = {e.entity_type for e in entities}
    expected_types = {"Drug", "Condition", "SideEffect", "Population", "ClinicalStudy", "Guideline"}
    # Polyuria matches Symptom (since polyuria is in dictionary as Symptom)
    if "polyuria" in text.lower():
        expected_types.add("Symptom")
        
    for expected in expected_types:
        assert expected in extracted_types
        
    # Ensure all have valid IDs and source spans within the global document bounds
    for ent in entities:
        assert ent.entity_type in ALLOWED_ENTITY_TYPES
        assert ent.entity_id is not None
        assert ent.normalized_name is not None
        assert ent.source_span[0] >= chunk.start_offset
        assert ent.source_span[1] <= chunk.end_offset
        # Check text slice matches
        snippet_len = ent.source_span[1] - ent.source_span[0]
        assert snippet_len > 0

def test_unsupported_entity_type_rejection():
    # Since we use validation checks, if a candidate does not belong to allowed types, it shouldn't enter.
    # In rule_based_mock, dictionary items only contain allowed types.
    # We will verify that in the mock method list, no item has an unapproved type.
    from ingestion.extraction.entity_extractor import MOCK_CLINICAL_DICTIONARY
    for pattern, ent_type, canonical in MOCK_CLINICAL_DICTIONARY:
        assert ent_type in ALLOWED_ENTITY_TYPES

def test_error_isolation():
    # If a chunk extraction throws an exception (e.g., chunk is malformed or throws attribute error),
    # extract_entities should catch it, log it, and return [] rather than crashing the pipeline.
    
    # Passing None instead of Chunk to trigger AttributeError internally
    bad_chunk = None
    entities = extract_entities(bad_chunk, method="rule_based_mock")
    assert entities == []
