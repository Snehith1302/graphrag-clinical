"""
Entity extraction module for clinical text chunks.
Supports rule-based mock matching (for test stability) and structured LLM calls.
"""
import re
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import httpx
from backend.app.config import settings
from backend.app.models.schemas import Chunk, Entity

logger = logging.getLogger("graphrag.ingestion.entity_extractor")

ALLOWED_ENTITY_TYPES = {
    "Drug", "Condition", "Symptom", "SideEffect", 
    "Population", "ClinicalStudy", "Guideline"
}

# Synonyms mapping for deterministic rule-based mock mode
MOCK_CLINICAL_DICTIONARY = [
    # Drugs
    (r"(?i)\bmetformin\s+hydrochloride\s+tablets?\b|\bmetformin\s+hydrochloride\b|\bmetformin\s+hcl\b|\bmetformin\b", "Drug", "Metformin"),
    (r"(?i)\binsulin\b", "Drug", "Insulin"),
    
    # Conditions
    (r"(?i)\btype\s+2\s+diabetes\s+mellitus\b|\btype\s+2\s+diabetes\b|\bdiabetes\s+mellitus\b|\bdiabetes\b", "Condition", "Type 2 Diabetes"),
    (r"(?i)\bdiabetic\s+ketoacidosis\b|\bketoacidosis\b", "Condition", "Diabetic Ketoacidosis"),
    (r"(?i)\blactic\s+acidosis\b", "Condition", "Lactic Acidosis"),
    (r"(?i)\bacidosis\b", "Condition", "Acidosis"),
    (r"(?i)\bhypertension\b", "Condition", "Hypertension"),
    
    # Side Effects / Symptoms
    (r"(?i)\bdiarrhea\b|\bdiarrhoea\b", "SideEffect", "Diarrhea"),
    (r"(?i)\bnausea\b", "SideEffect", "Nausea"),
    (r"(?i)\bvomiting\b", "SideEffect", "Vomiting"),
    (r"(?i)\bflatulence\b", "SideEffect", "Flatulence"),
    (r"(?i)\bindigestion\b", "SideEffect", "Indigestion"),
    (r"(?i)\babdominal\s+discomfort\b", "SideEffect", "Abdominal Discomfort"),
    (r"(?i)\bheadache\b", "SideEffect", "Headache"),
    (r"(?i)\basthenia\b", "SideEffect", "Asthenia"),
    (r"(?i)\bpolyuria\b", "Symptom", "Polyuria"),
    
    # Populations
    (r"(?i)\bsevere\s+renal\s+impairment\b", "Population", "Severe Renal Impairment"),
    (r"(?i)\brenal\s+impairment\b|\brenal\s+dysfunction\b|\brenal\s+insufficiency\b", "Population", "Renal Impairment"),
    (r"(?i)\bthird\s+trimester\s+pregnancy\b|\bpregnancy\b", "Population", "Pregnancy"),
    (r"(?i)\badults\b", "Population", "Adults"),
    
    # Studies
    (r"(?i)\befficacy\s+and\s+side\s+effects\s+of\s+metformin\s+in\s+type\s+2\s+diabetes\b", "ClinicalStudy", "Efficacy and Side Effects of Metformin in Type 2 Diabetes"),
    
    # Guidelines
    (r"(?i)\bmetformin\s+clinical\s+guideline\s+2024\b", "Guideline", "Metformin Clinical Guideline 2024")
]

def normalize_name(name: str, entity_type: str) -> str:
    """
    Standardizes raw entity mentions into canonical, title-cased forms.
    """
    n = name.strip().lower()
    
    if entity_type == "Drug":
        if "metformin" in n:
            return "Metformin"
        if "insulin" in n:
            return "Insulin"
            
    if entity_type == "Condition":
        if "type 2 diabetes" in n or "type ii diabetes" in n or "diabetes mellitus" in n:
            return "Type 2 Diabetes"
        if "lactic acidosis" in n:
            return "Lactic Acidosis"
        if "ketoacidosis" in n:
            return "Diabetic Ketoacidosis"
        if "acidosis" in n:
            return "Acidosis"
            
    if entity_type == "Population":
        if "severe renal" in n:
            return "Severe Renal Impairment"
        if "renal" in n:
            return "Renal Impairment"
        if "pregnancy" in n:
            return "Pregnancy"
            
    # Default to title-casing the cleaned name
    return name.strip().title()

def generate_entity_id(entity_type: str, normalized_name: str) -> str:
    """
    Generates a unique, stable, lowercase string identifier.
    Example: 'Drug' + 'Metformin Hydrochloride' -> 'drug_metformin_hydrochloride'
    """
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", normalized_name)
    slug = clean.strip().lower().replace(" ", "_")
    return f"{entity_type.lower()}_{slug}"

def extract_entities_mock(chunk: Chunk) -> List[Entity]:
    """
    Performs deterministic, dictionary-based text scanning.
    Useful for unit testing and offline development.
    """
    entities = []
    text = chunk.text
    
    for pattern, entity_type, canonical_name in MOCK_CLINICAL_DICTIONARY:
        # Validate that the type is allowed
        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue
            
        # Find all occurrences of the term in chunk text
        for match in re.finditer(pattern, text):
            matched_text = match.group(0)
            start_char = match.start()
            end_char = match.end()
            
            # Global character offsets
            global_start = chunk.start_offset + start_char
            global_end = chunk.start_offset + end_char
            
            norm_name = normalize_name(matched_text, entity_type)
            ent_id = generate_entity_id(entity_type, norm_name)
            
            # Check for duplicates within this chunk list to prevent overlaps of the same entity
            if any(e.entity_id == ent_id and e.source_span == (global_start, global_end) for e in entities):
                continue
                
            entities.append(Entity(
                entity_id=ent_id,
                normalized_name=norm_name,
                entity_type=entity_type,
                confidence=0.95,
                document_id=chunk.document_id,
                source_span=(global_start, global_end)
            ))
            
    return entities

def extract_entities_llm(chunk: Chunk) -> List[Entity]:
    """
    Calls the configured LLM API to extract clinical entities using structured outputs.
    """
    # System prompt enforcing strict output rules and types
    prompt = f"""You are a clinical NLP extractor. Your task is to extract entities from the provided clinical text.
Extract ONLY entities that belong to the following ALLOWED_TYPES:
{', '.join(ALLOWED_ENTITY_TYPES)}

Do not extract doctors, organizations, manufacturers, or generic terms that do not fit the ALLOWED_TYPES.

For each extracted entity, output:
1. entity_text: The exact text snippet as it appears in the text.
2. normalized_name: A standardized canonical name.
3. entity_type: One of the ALLOWED_TYPES.
4. confidence: A float between 0.0 and 1.0.

Return the output as a valid JSON object with the following schema:
{{
  "entities": [
    {{
      "entity_text": "metformin hcl",
      "normalized_name": "Metformin",
      "entity_type": "Drug",
      "confidence": 0.98
    }}
  ]
}}

TEXT TO EXTRACT:
\"\"\"
{chunk.text}
\"\"\"
"""
    headers = {
        "Content-Type": "application/json"
    }

    # Determine if calling OpenAI or Gemini based on configuration
    if "gemini" in settings.LLM_MODEL_NAME.lower():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL_NAME}:generateContent?key={settings.LLM_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
    else:
        # Default OpenAI format
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a precise clinical extraction assistant."},
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
                logger.warning(f"Rate limited (429) on entity extraction. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited (429) on entity extraction. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
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
        
        # Parse text field based on provider schema
        if "gemini" in settings.LLM_MODEL_NAME.lower():
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raw_text = res_json["choices"][0]["message"]["content"]
            
        extracted_data = json.loads(raw_text)
        entities_list = extracted_data.get("entities", [])
        
        entities = []
        for item in entities_list:
            entity_text = item.get("entity_text", "").strip()
            ent_type = item.get("entity_type", "").strip()
            confidence = float(item.get("confidence", 0.8))
            
            # Check constraints
            if not entity_text or ent_type not in ALLOWED_ENTITY_TYPES:
                logger.warning(f"Rejected unsupported or malformed entity: {item}")
                continue
                
            # Find char spans in chunk text
            # Fallback to case-insensitive find if direct match fails
            start_char = chunk.text.find(entity_text)
            if start_char == -1:
                start_char = chunk.text.lower().find(entity_text.lower())
                
            if start_char == -1:
                # If we cannot locate the text, skip span alignment or set to default (0, len)
                logger.warning(f"Could not locate entity text '{entity_text}' in chunk text. Skipping.")
                continue
                
            end_char = start_char + len(entity_text)
            global_start = chunk.start_offset + start_char
            global_end = chunk.start_offset + end_char
            
            norm_name = normalize_name(item.get("normalized_name", entity_text), ent_type)
            ent_id = generate_entity_id(ent_type, norm_name)
            
            entities.append(Entity(
                entity_id=ent_id,
                normalized_name=norm_name,
                entity_type=ent_type,
                confidence=confidence,
                document_id=chunk.document_id,
                source_span=(global_start, global_end)
            ))
            
        return entities
    except Exception as e:
        logger.error(f"LLM API extraction failed: {str(e)}")
        raise e

def extract_entities(chunk: Chunk, method: Optional[str] = None) -> List[Entity]:
    """
    Extracts clinical entities from a document text chunk.
    - Error isolation: exceptions are caught per chunk, logging the error and returning an empty list,
      preventing a single failure from stopping the whole ingestion pipeline.
    """
    try:
        if chunk is None:
            raise ValueError("Chunk object is None")
            
        logger.info(f"Extracting entities from chunk: {chunk.chunk_id}")
        
        # Fallback/Selection hierarchy
        if method is None:
            # If API key is mock or default, force local rule_based mode
            if settings.LLM_API_KEY in ["mock_key", "your_api_key_here", ""] or not settings.LLM_API_KEY:
                method = "rule_based_mock"
            else:
                method = "direct_llm"
                
        if method == "rule_based_mock":
            return extract_entities_mock(chunk)
        elif method == "direct_llm":
            return extract_entities_llm(chunk)
        else:
            logger.warning(f"Unknown extraction method '{method}'. Falling back to rule-based mock.")
            return extract_entities_mock(chunk)
    except Exception as e:
        # Section 8 Error Handling: Failed entity extraction (per chunk) -> Log, skip chunk, continue pipeline.
        chunk_id = chunk.chunk_id if chunk else "None"
        logger.error(f"Failed entity extraction for chunk {chunk_id}: {str(e)}. Continuing pipeline.")
        return []
