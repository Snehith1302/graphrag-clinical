"""
Data validation and deduplication module.
Validates clinical entity and relationship objects before graph database storage.
"""
import os
import logging
from typing import List, Tuple, Dict, Set
from backend.app.config import settings
from backend.app.models.schemas import Entity, Relationship
from ingestion.extraction.entity_extractor import ALLOWED_ENTITY_TYPES, normalize_name, generate_entity_id
from ingestion.extraction.relation_extractor import ALLOWED_RELATION_TYPES, ALLOWED_MAPPINGS, validate_relationship_types, check_contradictions

logger = logging.getLogger("graphrag.ingestion.validator")

# Ensure the logs folder exists to write validation logs
VALIDATION_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "data"
)
VALIDATION_LOG_FILE = os.path.join(VALIDATION_LOG_DIR, "validation_errors.log")

def write_validation_error(message: str) -> None:
    """
    Writes structured validation logs to a dedicated file for post-ingestion review.
    """
    try:
        os.makedirs(VALIDATION_LOG_DIR, exist_ok=True)
        with open(VALIDATION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{logging.getLevelName(logging.WARNING)}] {message}\n")
    except Exception as e:
        logger.error(f"Failed writing validation log: {str(e)}")

def validate_extracted_data(entities: List[Entity], relations: List[Relationship]) -> Tuple[List[Entity], List[Relationship]]:
    """
    Validates, normalizes, and deduplicates extracted entities and relationships.
    Preserves document provenance by merging source IDs and logs contradictions.
    Returns:
        Tuple[List[Entity], List[Relationship]]: Cleaned lists of unique entities and relationships.
    """
    logger.info("Initializing validation and deduplication layer...")
    
    # Check for empty lists early
    if not entities and not relations:
        logger.info("Empty extraction input lists passed to validator.")
        return [], []

    confidence_threshold = settings.RELATION_CONFIDENCE_THRESHOLD

    # ----------------------------------------------------
    # 1. Entity Validation and Deduplication
    # ----------------------------------------------------
    unique_entities: Dict[str, Entity] = {}
    old_to_new_id_map: Dict[str, str] = {}

    for ent in entities:
        # A. Enforce closed entity types
        if ent.entity_type not in ALLOWED_ENTITY_TYPES:
            msg = f"REJECTED ENTITY: Type '{ent.entity_type}' of entity '{ent.normalized_name}' is not in allowed types list."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        # B. Apply confidence threshold
        if ent.confidence < confidence_threshold:
            msg = f"REJECTED ENTITY: Confidence {ent.confidence} of '{ent.normalized_name}' is below threshold {confidence_threshold}."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        # C. Consistent name normalization and ID stability
        norm_name = normalize_name(ent.normalized_name, ent.entity_type)
        ent_id = generate_entity_id(ent.entity_type, norm_name)
        
        # Keep track of ID mapping from raw extraction ID to normalized ID
        old_to_new_id_map[ent.entity_id] = ent_id
        
        cleaned_entity = Entity(
            entity_id=ent_id,
            normalized_name=norm_name,
            entity_type=ent.entity_type,
            confidence=ent.confidence,
            document_id=ent.document_id,
            source_span=ent.source_span
        )

        # D. Deduplicate equivalent entity mentions
        if ent_id in unique_entities:
            # Entity already exists, keep the instance with the higher confidence score
            existing_ent = unique_entities[ent_id]
            if cleaned_entity.confidence > existing_ent.confidence:
                unique_entities[ent_id] = cleaned_entity
        else:
            unique_entities[ent_id] = cleaned_entity

    # ----------------------------------------------------
    # 2. Relationship Validation, Deduplication & Provenance Merging
    # ----------------------------------------------------
    unique_relationships: Dict[str, Relationship] = {}

    for rel in relations:
        # Resolve source/target entity IDs to their canonical normalized values
        source_id = old_to_new_id_map.get(rel.source_entity_id, rel.source_entity_id)
        target_id = old_to_new_id_map.get(rel.target_entity_id, rel.target_entity_id)

        # A. Enforce closed relationship types
        if rel.relation_type not in ALLOWED_RELATION_TYPES:
            msg = f"REJECTED RELATIONSHIP: Relationship type '{rel.relation_type}' between '{source_id}' and '{target_id}' is not in allowed relationships list."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        # B. Check grounding: Reject if source/target entities do not exist in unique_entities list
        if source_id not in unique_entities:
            msg = f"REJECTED RELATIONSHIP: Source entity ID '{source_id}' does not exist in validated entity ledger."
            logger.warning(msg)
            write_validation_error(msg)
            continue
            
        if target_id not in unique_entities:
            msg = f"REJECTED RELATIONSHIP: Target entity ID '{target_id}' does not exist in validated entity ledger."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        source_ent = unique_entities[source_id]
        target_ent = unique_entities[target_id]

        # C. Enforce valid source -> relationship -> target type mappings
        if not validate_relationship_types(source_ent.entity_type, rel.relation_type, target_ent.entity_type):
            msg = f"REJECTED RELATIONSHIP: Type incompatibility {source_ent.entity_type} -> {rel.relation_type} -> {target_ent.entity_type} (source: '{source_ent.normalized_name}', target: '{target_ent.normalized_name}')."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        # D. Reject if empty provenance / source_ids list
        if not rel.source_ids or all(s.strip() == "" for s in rel.source_ids):
            msg = f"REJECTED RELATIONSHIP: Provenance source_ids is empty for relation {rel.relation_id}."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        # E. Apply confidence threshold
        if rel.confidence < confidence_threshold:
            msg = f"REJECTED RELATIONSHIP: Confidence {rel.confidence} of relation '{rel.relation_id}' is below threshold {confidence_threshold}."
            logger.warning(msg)
            write_validation_error(msg)
            continue

        # F. Deduplicate relationships and merge provenance (source_ids)
        rel_id = f"{source_id}_{rel.relation_type.lower()}_{target_id}"
        
        # Clean clean_ids and deduplicate source reference entries
        clean_source_ids = list(set([s.strip() for s in rel.source_ids if s.strip()]))
        
        if rel_id in unique_relationships:
            existing_rel = unique_relationships[rel_id]
            # Merge source documents list (provenance merging)
            merged_sources = list(set(existing_rel.source_ids + clean_source_ids))
            
            # Preserve highest confidence score
            best_confidence = max(existing_rel.confidence, rel.confidence)
            
            unique_relationships[rel_id] = Relationship(
                relation_id=rel_id,
                source_entity_id=source_id,
                relation_type=rel.relation_type,
                target_entity_id=target_id,
                confidence=best_confidence,
                source_ids=merged_sources
            )
        else:
            unique_relationships[rel_id] = Relationship(
                relation_id=rel_id,
                source_entity_id=source_id,
                relation_type=rel.relation_type,
                target_entity_id=target_id,
                confidence=rel.confidence,
                source_ids=clean_source_ids
            )

    validated_entities = list(unique_entities.values())
    validated_relations = list(unique_relationships.values())

    # ----------------------------------------------------
    # 3. Detect and Flag Contradictory Relationships
    # ----------------------------------------------------
    contradictions = check_contradictions(validated_relations)
    for conflict in contradictions:
        write_validation_error(conflict)

    logger.info(f"Validation layer complete: {len(validated_entities)} unique entities and {len(validated_relations)} unique relationships validated.")
    return validated_entities, validated_relations
