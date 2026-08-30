"""
Unit tests for clinical relation extraction and validation.
"""
from backend.app.models.schemas import Chunk, Entity, Relationship
from ingestion.extraction.relation_extractor import (
    extract_relations,
    validate_relationship_types,
    generate_relation_id,
    check_contradictions,
    ALLOWED_RELATION_TYPES
)

def test_relationship_type_and_mappings_validation():
    # 1. Valid Mappings (Source -> Relation -> Target)
    assert validate_relationship_types("Drug", "TREATS", "Condition") is True
    assert validate_relationship_types("Drug", "CAUSES", "SideEffect") is True
    assert validate_relationship_types("Condition", "HAS_SYMPTOM", "Symptom") is True
    assert validate_relationship_types("Drug", "INTERACTS_WITH", "Drug") is True
    assert validate_relationship_types("Drug", "CONTRAINDICATED_FOR", "Condition") is True
    assert validate_relationship_types("Drug", "CONTRAINDICATED_FOR", "Population") is True
    assert validate_relationship_types("Guideline", "RECOMMENDS", "Drug") is True

    # 2. Invalid relation type
    assert validate_relationship_types("Drug", "MANUFACTURES", "Condition") is False
    assert validate_relationship_types("Drug", "INJECTS", "SideEffect") is False

    # 3. Invalid entity types compatibility
    assert validate_relationship_types("Symptom", "TREATS", "Drug") is False
    assert validate_relationship_types("SideEffect", "CAUSES", "Drug") is False
    assert validate_relationship_types("Drug", "HAS_SYMPTOM", "Condition") is False

def test_generate_relation_id():
    assert generate_relation_id("drug_metformin", "TREATS", "condition_type_2_diabetes") == "drug_metformin_treats_condition_type_2_diabetes"

def test_extract_all_valid_relationships():
    # Construct a set of mock entities
    entities = [
        Entity(entity_id="guideline_metformin_clinical_guideline_2024", normalized_name="Metformin Clinical Guideline 2024", entity_type="Guideline", confidence=1.0, document_id="doc_1", source_span=(0, 10)),
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=1.0, document_id="doc_1", source_span=(12, 21)),
        Entity(entity_id="condition_type_2_diabetes", normalized_name="Type 2 Diabetes", entity_type="Condition", confidence=1.0, document_id="doc_1", source_span=(23, 38)),
        Entity(entity_id="sideeffect_diarrhea", normalized_name="Diarrhea", entity_type="SideEffect", confidence=1.0, document_id="doc_1", source_span=(40, 48)),
        Entity(entity_id="population_severe_renal_impairment", normalized_name="Severe Renal Impairment", entity_type="Population", confidence=1.0, document_id="doc_1", source_span=(50, 73)),
        Entity(entity_id="drug_insulin", normalized_name="Insulin", entity_type="Drug", confidence=1.0, document_id="doc_1", source_span=(75, 82)),
        Entity(entity_id="symptom_polyuria", normalized_name="Polyuria", entity_type="Symptom", confidence=1.0, document_id="doc_1", source_span=(84, 92))
    ]
    
    chunk = Chunk(
        chunk_id="chunk_test_relations",
        document_id="doc_1",
        text="Sample text containing Metformin Clinical Guideline 2024, Metformin, Type 2 Diabetes, diarrhea, severe renal impairment, Insulin, and polyuria.",
        start_offset=0,
        end_offset=150
    )
    
    relations = extract_relations(chunk, entities, method="rule_based_mock")
    assert len(relations) > 0
    
    rel_types = {r.relation_type for r in relations}
    
    # Assert every expected relation was created:
    # 1. TREATS: drug_metformin treats condition_type_2_diabetes
    # 2. CAUSES: drug_metformin causes sideeffect_diarrhea
    # 3. CONTRAINDICATED_FOR: drug_metformin contraindicated_for population_severe_renal_impairment
    # 4. INTERACTS_WITH: drug_metformin interacts_with drug_insulin
    # 5. RECOMMENDS: guideline_metformin_clinical_guideline_2024 recommends drug_metformin
    # 6. HAS_SYMPTOM: condition_type_2_diabetes has_symptom symptom_polyuria
    expected_relations = {"TREATS", "CAUSES", "CONTRAINDICATED_FOR", "INTERACTS_WITH", "RECOMMENDS", "HAS_SYMPTOM"}
    for expected in expected_relations:
        assert expected in rel_types
        
    for r in relations:
        assert r.relation_id == generate_relation_id(r.source_entity_id, r.relation_type, r.target_entity_id)
        assert len(r.source_ids) > 0
        assert r.source_ids == ["doc_1"]
        assert r.confidence >= 0.6

def test_contradictory_relationship_flagging():
    # If a list has both TREATS and CONTRAINDICATED_FOR between the same drug and condition
    relations = [
        Relationship(
            relation_id="drug_metformin_treats_condition_type_2_diabetes",
            source_entity_id="drug_metformin",
            relation_type="TREATS",
            target_entity_id="condition_type_2_diabetes",
            confidence=0.9,
            source_ids=["doc_1"]
        ),
        Relationship(
            relation_id="drug_metformin_contraindicated_for_condition_type_2_diabetes",
            source_entity_id="drug_metformin",
            relation_type="CONTRAINDICATED_FOR",
            target_entity_id="condition_type_2_diabetes",
            confidence=0.9,
            source_ids=["doc_1"]
        )
    ]
    
    flagged = check_contradictions(relations)
    assert len(flagged) == 1
    assert "CONTRADICTION DETECTED" in flagged[0]
    assert "drug_metformin" in flagged[0]
    assert "condition_type_2_diabetes" in flagged[0]

def test_error_isolation():
    # Ensure error isolation catches issues and returns empty lists rather than crashing
    bad_chunk = None
    entities = []
    relations = extract_relations(bad_chunk, entities, method="rule_based_mock")
    assert relations == []
