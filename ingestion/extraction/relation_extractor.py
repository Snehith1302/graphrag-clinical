"""
Relation extraction module for clinical text chunks.
Supports rule-based mock matching (for test stability) and structured LLM calls.
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import httpx
from backend.app.config import settings
from backend.app.models.schemas import Chunk, Entity, Relationship

logger = logging.getLogger("graphrag.ingestion.relation_extractor")

ALLOWED_RELATION_TYPES = {
    "TREATS", "CAUSES", "HAS_SYMPTOM", "INTERACTS_WITH", 
    "CONTRAINDICATED_FOR", "RECOMMENDS"
}

# Source -> Target type compatibility constraints (Section 6)
ALLOWED_MAPPINGS = {
    "TREATS": [("Drug", "Condition")],
    "CAUSES": [("Drug", "SideEffect")],
    "HAS_SYMPTOM": [("Condition", "Symptom")],
    "INTERACTS_WITH": [("Drug", "Drug")],
    "CONTRAINDICATED_FOR": [("Drug", "Condition"), ("Drug", "Population")],
    "RECOMMENDS": [("Guideline", "Drug")]
}

def generate_relation_id(source_id: str, rel_type: str, target_id: str) -> str:
    """
    Generates a unique, stable, lowercase string identifier for relationships.
    """
    return f"{source_id}_{rel_type.lower()}_{target_id}"

def validate_relationship_types(source_type: str, rel_type: str, target_type: str) -> bool:
    """
    Validates if a source-relation-target type triple is allowed by the schema.
    """
    if rel_type not in ALLOWED_RELATION_TYPES:
        return False
    
    valid_pairs = ALLOWED_MAPPINGS.get(rel_type, [])
    return (source_type, target_type) in valid_pairs

def check_contradictions(relations: List[Relationship]) -> List[str]:
    """
    Scans a list of relationships for contradictions (e.g. TREATS and CONTRAINDICATED_FOR
    between the same drug and condition).
    Logs contradictions and returns list of flagged messages.
    """
    contradictions = []
    # Create lookup map
    treats_map = set()
    contra_map = set()

    for r in relations:
        if r.relation_type == "TREATS":
            treats_map.add((r.source_entity_id, r.target_entity_id))
        elif r.relation_type == "CONTRAINDICATED_FOR":
            contra_map.add((r.source_entity_id, r.target_entity_id))

    # Identify intersection
    conflicts = treats_map.intersection(contra_map)
    for source, target in conflicts:
        msg = f"CONTRADICTION DETECTED: Entity '{source}' both TREATS and is CONTRAINDICATED_FOR '{target}'."
        logger.warning(msg)
        contradictions.append(msg)

    return contradictions

def extract_relations_mock(chunk: Chunk, entities: List[Entity]) -> List[Relationship]:
    """
    Deterministic rule-based mock extractor.
    Creates relationships between entities that co-occur in the same chunk
    if they match established clinical rules.
    """
    relations = []
    entity_map = {e.entity_id: e for e in entities}
    
    # Check all entity pairs in this chunk
    ent_ids = list(entity_map.keys())
    for i in range(len(ent_ids)):
        for j in range(len(ent_ids)):
            if i == j:
                continue
                
            source_id = ent_ids[i]
            target_id = ent_ids[j]
            source = entity_map[source_id]
            target = entity_map[target_id]
            
            # Clinical Facts Inference Rules based on Mock Dictionary
            rel_type = None
            
            # 1. TREATS: Metformin treats Type 2 Diabetes
            if source.normalized_name == "Metformin" and target.normalized_name == "Type 2 Diabetes":
                rel_type = "TREATS"
                
            # 2. CAUSES: Metformin causes SideEffects (diarrhea, nausea, vomiting, etc.)
            elif source.normalized_name == "Metformin" and target.entity_type == "SideEffect":
                rel_type = "CAUSES"
                
            # 3. CONTRAINDICATED_FOR: Metformin is contraindicated in Severe Renal Impairment
            elif source.normalized_name == "Metformin" and target.normalized_name in ["Severe Renal Impairment", "Pregnancy"]:
                rel_type = "CONTRAINDICATED_FOR"
                
            # 4. INTERACTS_WITH: Metformin interacts with Insulin
            elif source.normalized_name == "Metformin" and target.normalized_name == "Insulin":
                rel_type = "INTERACTS_WITH"
                
            # 5. HAS_SYMPTOM: Type 2 Diabetes has symptom Polyuria
            elif source.normalized_name == "Type 2 Diabetes" and target.normalized_name == "Polyuria":
                rel_type = "HAS_SYMPTOM"
                
            # 6. RECOMMENDS: Metformin Clinical Guideline 2024 recommends Metformin
            elif source.normalized_name == "Metformin Clinical Guideline 2024" and target.normalized_name == "Metformin":
                rel_type = "RECOMMENDS"
                
            if rel_type:
                # Validate source/target type compatibility
                if not validate_relationship_types(source.entity_type, rel_type, target.entity_type):
                    continue
                    
                rel_id = generate_relation_id(source_id, rel_type, target_id)
                relations.append(Relationship(
                    relation_id=rel_id,
                    source_entity_id=source_id,
                    relation_type=rel_type,
                    target_entity_id=target_id,
                    confidence=0.95,
                    source_ids=[chunk.document_id]  # Provenance
                ))
                
    return relations

def extract_relations_llm(chunk: Chunk, entities: List[Entity]) -> List[Relationship]:
    """
    Invokes the LLM to extract relationships from the chunk text based on the detected entities.
    """
    entity_map = {e.entity_id: e for e in entities}
    entities_details = [
        {"entity_id": e.entity_id, "name": e.normalized_name, "type": e.entity_type}
        for e in entities
    ]

    prompt = f"""You are a clinical NLP extractor. Your task is to extract relationships between the provided entities in the clinical text.
You must ONLY output the following allowed relation types:
{', '.join(ALLOWED_RELATION_TYPES)}

Type Constraints:
- TREATS: Drug -> Condition
- CAUSES: Drug -> SideEffect
- HAS_SYMPTOM: Condition -> Symptom
- INTERACTS_WITH: Drug -> Drug
- CONTRAINDICATED_FOR: Drug -> Condition OR Drug -> Population
- RECOMMENDS: Guideline -> Drug

Here are the entities found in this text:
{json.dumps(entities_details, indent=2)}

Do not extract relationships involving any other entity IDs.

Return the output as a valid JSON object with the following schema:
{{
  "relationships": [
    {{
      "source_entity_id": "drug_metformin",
      "relation_type": "TREATS",
      "target_entity_id": "condition_type_2_diabetes",
      "confidence": 0.92
    }}
  ]
}}

TEXT:
\"\"\"
{chunk.text}
\"\"\"
"""
    headers = {"Content-Type": "application/json"}
    
    if "gemini" in settings.LLM_MODEL_NAME.lower():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL_NAME}:generateContent?key={settings.LLM_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a precise clinical relation extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

    import time
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "1"))
    backoff = float(os.environ.get("LLM_BACKOFF_SECONDS", "4.0"))
    response = None
    
    for attempt in range(max_retries):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            if response.status_code == 429:
                logger.warning(f"Rate limited (429) on relation extraction. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited (429) on relation extraction. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise e
        except httpx.RequestError as e:
            logger.warning(f"Request error: {str(e)}. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 1.5
            continue

    if not response or response.status_code == 429:
        raise Exception("API rate limit exceeded after maximum retries.")

    try:
        res_json = response.json()
        
        if "gemini" in settings.LLM_MODEL_NAME.lower():
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raw_text = res_json["choices"][0]["message"]["content"]
            
        extracted_data = json.loads(raw_text)
        relations_list = extracted_data.get("relationships", [])
        
        relations = []
        for item in relations_list:
            src_id = item.get("source_entity_id", "").strip()
            tgt_id = item.get("target_entity_id", "").strip()
            rel_type = item.get("relation_type", "").strip()
            confidence = float(item.get("confidence", 0.8))
            
            # Grounding check: Source/Target entities must exist in entities list
            if src_id not in entity_map or tgt_id not in entity_map:
                logger.warning(f"Rejected relation: entities '{src_id}' or '{tgt_id}' not found in chunk entity list.")
                continue
                
            source_entity = entity_map[src_id]
            target_entity = entity_map[tgt_id]
            
            # Validation checks
            if rel_type not in ALLOWED_RELATION_TYPES:
                logger.warning(f"Rejected relation: type '{rel_type}' is not allowed.")
                continue
                
            # Type compatibility constraint validation
            if not validate_relationship_types(source_entity.entity_type, rel_type, target_entity.entity_type):
                logger.warning(f"Rejected relation: type mismatch between {source_entity.entity_type} -> {rel_type} -> {target_entity.entity_type}")
                continue
                
            # Confidence threshold validation
            if confidence < settings.RELATION_CONFIDENCE_THRESHOLD:
                logger.warning(f"Rejected relation: confidence {confidence} below threshold {settings.RELATION_CONFIDENCE_THRESHOLD}")
                continue
                
            rel_id = generate_relation_id(src_id, rel_type, tgt_id)
            relations.append(Relationship(
                relation_id=rel_id,
                source_entity_id=src_id,
                relation_type=rel_type,
                target_entity_id=tgt_id,
                confidence=confidence,
                source_ids=[chunk.document_id]  # Provenance
            ))
            
        return relations
    except Exception as e:
        logger.error(f"LLM API relation extraction failed: {str(e)}")
        raise e

def extract_relations(chunk: Chunk, entities: List[Entity], method: Optional[str] = None) -> List[Relationship]:
    """
    Extracts clinical relationships from a document text chunk given its entities.
    - Error isolation: exceptions are caught per chunk, logging the error and returning an empty list,
      preventing a single failure from stopping the whole ingestion pipeline.
    """
    try:
        if chunk is None:
            raise ValueError("Chunk object is None")
            
        logger.info(f"Extracting relationships for chunk: {chunk.chunk_id}")
        
        if not entities:
            # Cannot have relations without entities
            return []
            
        # Fallback/Selection hierarchy
        if method is None:
            # If API key is mock or default, force local rule_based mode
            if settings.LLM_API_KEY in ["mock_key", "your_api_key_here", ""] or not settings.LLM_API_KEY:
                method = "rule_based_mock"
            else:
                method = "direct_llm"
                
        if method == "rule_based_mock":
            extracted = extract_relations_mock(chunk, entities)
        elif method == "direct_llm":
            extracted = extract_relations_llm(chunk, entities)
        else:
            logger.warning(f"Unknown extraction method '{method}'. Falling back to rule-based mock.")
            extracted = extract_relations_mock(chunk, entities)
            
        # Check and flag contradictory relations
        check_contradictions(extracted)
        
        return extracted
    except Exception as e:
        # Section 8 Error Handling: Failed relation extraction (per chunk) -> Log, skip chunk, continue pipeline.
        chunk_id = chunk.chunk_id if chunk else "None"
        logger.error(f"Failed relation extraction for chunk {chunk_id}: {str(e)}. Continuing pipeline.")
        return []
