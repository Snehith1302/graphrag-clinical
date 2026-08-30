"""
Unit tests for data validation, deduplication, and conflict flagging.
"""
from backend.app.models.schemas import Entity, Relationship
from ingestion.validation.validator import validate_extracted_data

def test_validate_empty_lists():
    ents, rels = validate_extracted_data([], [])
    assert ents == []
    assert rels == []

def test_validate_invalid_entity_type():
    entities = [
        # Invalid type "Doctor"
        Entity(entity_id="doctor_alice", normalized_name="Alice Smith", entity_type="Doctor", confidence=0.9, document_id="doc1", source_span=(0, 10))
    ]
    ents, rels = validate_extracted_data(entities, [])
    assert len(ents) == 0

def test_validate_low_confidence_rejection():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.3, document_id="doc1", source_span=(0, 10))
    ]
    ents, rels = validate_extracted_data(entities, [])
    assert len(ents) == 0

def test_validate_deduplicate_entities():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="metformin hcl", entity_type="Drug", confidence=0.7, document_id="doc1", source_span=(0, 10)),
        # Duplicate with higher confidence
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.95, document_id="doc2", source_span=(20, 29))
    ]
    ents, rels = validate_extracted_data(entities, [])
    assert len(ents) == 1
    assert ents[0].normalized_name == "Metformin"
    assert ents[0].confidence == 0.95

def test_validate_relation_type_and_grounding():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 10))
    ]
    
    relations = [
        # 1. Invalid relation type MANUFACTURES
        Relationship(relation_id="rel1", source_entity_id="drug_metformin", relation_type="MANUFACTURES", target_entity_id="condition_diabetes", confidence=0.9, source_ids=["doc1"]),
        # 2. Missing target entity grounding
        Relationship(relation_id="rel2", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.9, source_ids=["doc1"])
    ]
    
    ents, rels = validate_extracted_data(entities, relations)
    assert len(rels) == 0

def test_validate_relation_type_compatibility():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 10)),
        Entity(entity_id="sideeffect_diarrhea", normalized_name="Diarrhea", entity_type="SideEffect", confidence=0.9, document_id="doc1", source_span=(20, 30))
    ]
    
    relations = [
        # Invalid type mapping: TREATS can only go from Drug -> Condition, not Drug -> SideEffect
        Relationship(relation_id="rel1", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="sideeffect_diarrhea", confidence=0.9, source_ids=["doc1"])
    ]
    
    ents, rels = validate_extracted_data(entities, relations)
    assert len(rels) == 0

def test_validate_relation_missing_provenance():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 10)),
        Entity(entity_id="condition_diabetes", normalized_name="Type 2 Diabetes", entity_type="Condition", confidence=0.9, document_id="doc1", source_span=(20, 30))
    ]
    
    relations = [
        # Empty source_ids (provenance missing)
        Relationship(relation_id="rel1", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.9, source_ids=[])
    ]
    
    ents, rels = validate_extracted_data(entities, relations)
    assert len(rels) == 0

def test_validate_relation_deduplication_and_provenance_merge():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 10)),
        Entity(entity_id="condition_diabetes", normalized_name="Type 2 Diabetes", entity_type="Condition", confidence=0.9, document_id="doc1", source_span=(20, 30))
    ]
    
    relations = [
        Relationship(relation_id="r1", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.8, source_ids=["doc1"]),
        # Duplicate relation from another document with higher confidence
        Relationship(relation_id="r2", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.95, source_ids=["doc2", "doc1"])
    ]
    
    ents, rels = validate_extracted_data(entities, relations)
    assert len(rels) == 1
    assert rels[0].confidence == 0.95
    # Provenance list should be merged and deduplicated
    assert set(rels[0].source_ids) == {"doc1", "doc2"}

def test_validate_contradictory_relations():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 10)),
        Entity(entity_id="condition_diabetes", normalized_name="Type 2 Diabetes", entity_type="Condition", confidence=0.9, document_id="doc1", source_span=(20, 30))
    ]
    
    relations = [
        Relationship(relation_id="r1", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.9, source_ids=["doc1"]),
        # Contradiction: Metformin contraindicated for Type 2 Diabetes
        Relationship(relation_id="r2", source_entity_id="drug_metformin", relation_type="CONTRAINDICATED_FOR", target_entity_id="condition_diabetes", confidence=0.9, source_ids=["doc1"])
    ]
    
    ents, rels = validate_extracted_data(entities, relations)
    
    # Assert both relationships are preserved (no silent deletion)
    assert len(rels) == 2
    rel_types = {r.relation_type for r in rels}
    assert "TREATS" in rel_types
    assert "CONTRAINDICATED_FOR" in rel_types

def test_validation_deterministic_and_idempotent():
    entities = [
        Entity(entity_id="drug_metformin", normalized_name="Metformin", entity_type="Drug", confidence=0.9, document_id="doc1", source_span=(0, 10)),
        Entity(entity_id="condition_diabetes", normalized_name="Type 2 Diabetes", entity_type="Condition", confidence=0.9, document_id="doc1", source_span=(20, 30))
    ]
    
    relations = [
        Relationship(relation_id="r1", source_entity_id="drug_metformin", relation_type="TREATS", target_entity_id="condition_diabetes", confidence=0.8, source_ids=["doc1"])
    ]
    
    ents_run1, rels_run1 = validate_extracted_data(entities, relations)
    ents_run2, rels_run2 = validate_extracted_data(ents_run1, rels_run1)  # Idempotent pass
    
    assert len(ents_run1) == len(ents_run2)
    assert len(rels_run1) == len(rels_run2)
    assert ents_run1[0].entity_id == ents_run2[0].entity_id
    assert rels_run1[0].relation_id == rels_run2[0].relation_id
