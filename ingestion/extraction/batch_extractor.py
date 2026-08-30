"""
Batch extraction module for clinical text chunks.
Supports single-pass structured LLM requests per chunk batch and mock rule-based extraction for testing.
"""
import os
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import httpx

from backend.app.config import settings
from backend.app.models.schemas import Chunk, Entity, Relationship
from ingestion.extraction.entity_extractor import (
    ALLOWED_ENTITY_TYPES,
    normalize_name,
    generate_entity_id,
    extract_entities_mock
)
from ingestion.extraction.relation_extractor import (
    ALLOWED_RELATION_TYPES,
    ALLOWED_MAPPINGS,
    generate_relation_id,
    validate_relationship_types,
    check_contradictions,
    extract_relations_mock
)

logger = logging.getLogger("graphrag.ingestion.batch_extractor")

def parse_batch_response(
    raw_text: str,
    chunks: List[Chunk]
) -> Tuple[List[Entity], List[Relationship]]:
    """
    Parses and strictly validates the raw JSON response from the LLM batch extraction request.
    Maps results to input chunks by chunk_id, verifies entity character spans, entity type allowedness,
    relationship type constraints, and entity grounding.
    """
    extracted_data = json.loads(raw_text)
    chunks_data = extracted_data.get("chunks", [])
    
    chunk_map = {c.chunk_id: c for c in chunks}
    batch_entities: List[Entity] = []
    batch_relationships: List[Relationship] = []
    
    for chunk_item in chunks_data:
        chunk_id = chunk_item.get("chunk_id", "").strip()
        if not chunk_id or chunk_id not in chunk_map:
            logger.warning(f"Batch response contained unknown or missing chunk_id: '{chunk_id}'. Skipping chunk item.")
            continue
            
        chunk = chunk_map[chunk_id]
        chunk_entities: List[Entity] = []
        
        # 1. Parse and validate entities
        entities_list = chunk_item.get("entities", [])
        for item in entities_list:
            entity_text = item.get("entity_text", "").strip()
            ent_type = item.get("entity_type", "").strip()
            confidence = float(item.get("confidence", 0.8))
            
            if not entity_text or ent_type not in ALLOWED_ENTITY_TYPES:
                logger.warning(f"Rejected malformed or unallowed entity '{entity_text}' of type '{ent_type}' in chunk {chunk_id}.")
                continue
                
            # Locate character span in chunk text
            start_char = chunk.text.find(entity_text)
            if start_char == -1:
                start_char = chunk.text.lower().find(entity_text.lower())
                
            if start_char == -1:
                logger.warning(f"Could not locate entity text '{entity_text}' in chunk '{chunk_id}'. Skipping entity.")
                continue
                
            end_char = start_char + len(entity_text)
            global_start = chunk.start_offset + start_char
            global_end = chunk.start_offset + end_char
            
            norm_name = normalize_name(item.get("normalized_name", entity_text), ent_type)
            ent_id = generate_entity_id(ent_type, norm_name)
            
            # Deduplicate entities within this chunk
            if any(e.entity_id == ent_id and e.source_span == (global_start, global_end) for e in chunk_entities):
                continue
                
            chunk_entities.append(Entity(
                entity_id=ent_id,
                normalized_name=norm_name,
                entity_type=ent_type,
                confidence=confidence,
                document_id=chunk.document_id,
                source_span=(global_start, global_end)
            ))
            
        batch_entities.extend(chunk_entities)
        
        # Build lookup for relationship grounding
        entity_lookup = {}
        for e in chunk_entities:
            entity_lookup[e.entity_id] = e
            entity_lookup[e.normalized_name.lower()] = e
            start, end = e.source_span
            local_start = start - chunk.start_offset
            local_end = end - chunk.start_offset
            snippet = chunk.text[local_start:local_end].strip().lower()
            if snippet:
                entity_lookup[snippet] = e
                
        # 2. Parse and validate relationships
        relationships_list = chunk_item.get("relationships", [])
        chunk_relations: List[Relationship] = []
        for rel_item in relationships_list:
            src_ref = rel_item.get("source_entity_text", rel_item.get("source_entity_id", "")).strip().lower()
            tgt_ref = rel_item.get("target_entity_text", rel_item.get("target_entity_id", "")).strip().lower()
            rel_type = rel_item.get("relation_type", "").strip()
            confidence = float(rel_item.get("confidence", 0.8))
            
            # Grounding check: source and target must match extracted entities in this chunk
            source_entity = entity_lookup.get(src_ref)
            target_entity = entity_lookup.get(tgt_ref)
            
            if not source_entity or not target_entity:
                logger.warning(f"Rejected relationship in chunk {chunk_id}: ungrounded source '{src_ref}' or target '{tgt_ref}'.")
                continue
                
            # Type allowedness check
            if rel_type not in ALLOWED_RELATION_TYPES:
                logger.warning(f"Rejected relationship in chunk {chunk_id}: type '{rel_type}' is not allowed.")
                continue
                
            # Source -> Target type compatibility constraint check
            if not validate_relationship_types(source_entity.entity_type, rel_type, target_entity.entity_type):
                logger.warning(f"Rejected relationship in chunk {chunk_id}: incompatible type mapping {source_entity.entity_type} -> {rel_type} -> {target_entity.entity_type}.")
                continue
                
            # Confidence threshold check
            if confidence < settings.RELATION_CONFIDENCE_THRESHOLD:
                logger.warning(f"Rejected relationship in chunk {chunk_id}: confidence {confidence} below threshold {settings.RELATION_CONFIDENCE_THRESHOLD}.")
                continue
                
            rel_id = generate_relation_id(source_entity.entity_id, rel_type, target_entity.entity_id)
            
            # Deduplicate relationships within chunk
            if any(r.relation_id == rel_id for r in chunk_relations):
                continue
                
            chunk_relations.append(Relationship(
                relation_id=rel_id,
                source_entity_id=source_entity.entity_id,
                relation_type=rel_type,
                target_entity_id=target_entity.entity_id,
                confidence=confidence,
                source_ids=[chunk.document_id]  # Provenance
            ))
            
        check_contradictions(chunk_relations)
        batch_relationships.extend(chunk_relations)
        
    return batch_entities, batch_relationships

def extract_batch_mock(chunks: List[Chunk]) -> Tuple[List[Entity], List[Relationship], List[str], int]:
    """
    Deterministic rule-based batch extractor.
    Runs extract_entities_mock and extract_relations_mock per chunk.
    Returns (entities, relationships, failed_chunk_ids, num_429_retries).
    """
    all_entities: List[Entity] = []
    all_relations: List[Relationship] = []
    for chunk in chunks:
        ents = extract_entities_mock(chunk)
        rels = extract_relations_mock(chunk, ents)
        all_entities.extend(ents)
        all_relations.extend(rels)
    return all_entities, all_relations, [], 0

def extract_batch_llm(
    chunks: List[Chunk],
    max_retries: Optional[int] = None,
    backoff: Optional[float] = None
) -> Tuple[List[Entity], List[Relationship], List[str], int]:
    """
    Executes a single combined structured LLM API request for a batch of chunks.
    Treats HTTP 429 rate limits as authoritative and retries using exponential backoff.
    CRITICAL: Never falls back to mock/rule-based extraction if retries fail.
    Returns (entities, relationships, failed_chunk_ids, num_429_retries).
    """
    if max_retries is None:
        max_retries = settings.LLM_MAX_RETRIES
    if backoff is None:
        backoff = settings.LLM_BACKOFF_SECONDS

    chunks_payload = [
        {"chunk_id": c.chunk_id, "text": c.text}
        for c in chunks
    ]
    
    prompt = f"""You are a clinical NLP extractor. Your task is to extract clinical entities and relationships from the provided clinical text chunks in a single structured pass.

ALLOWED ENTITY TYPES:
{', '.join(sorted(ALLOWED_ENTITY_TYPES))}

ALLOWED RELATION TYPES AND CONSTRAINTS:
- TREATS: Drug -> Condition
- CAUSES: Drug -> SideEffect
- HAS_SYMPTOM: Condition -> Symptom
- INTERACTS_WITH: Drug -> Drug
- CONTRAINDICATED_FOR: Drug -> Condition OR Drug -> Population
- RECOMMENDS: Guideline -> Drug

For each chunk, extract the entities and the relationships connecting those entities.

Return the output as a valid JSON object matching this schema:
{{
  "chunks": [
    {{
      "chunk_id": "chunk_doc1_0",
      "entities": [
        {{
          "entity_text": "metformin hcl",
          "normalized_name": "Metformin",
          "entity_type": "Drug",
          "confidence": 0.95
        }}
      ],
      "relationships": [
        {{
          "source_entity_text": "metformin hcl",
          "relation_type": "TREATS",
          "target_entity_text": "type 2 diabetes",
          "confidence": 0.90
        }}
      ]
    }}
  ]
}}

CHUNKS TO PROCESS:
{json.dumps(chunks_payload, indent=2)}
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
                {"role": "system", "content": "You are a precise clinical extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

    response = None
    curr_backoff = backoff
    chunk_ids = [c.chunk_id for c in chunks]
    num_429_retries = 0

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Executing batch LLM extraction for {len(chunks)} chunks (Attempt {attempt}/{max_retries})...")
            response = httpx.post(url, json=payload, headers=headers, timeout=45.0)
            
            if response.status_code == 429:
                num_429_retries += 1
                logger.warning(f"Rate limited (HTTP 429) on batch extraction. Retrying in {curr_backoff}s... (Attempt {attempt}/{max_retries})")
                time.sleep(curr_backoff)
                curr_backoff *= 2.0
                continue
                
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                num_429_retries += 1
                logger.warning(f"Rate limited (HTTP 429) on batch extraction. Retrying in {curr_backoff}s... (Attempt {attempt}/{max_retries})")
                time.sleep(curr_backoff)
                curr_backoff *= 2.0
                continue
            logger.error(f"HTTP status error during batch extraction: {str(e)}")
            if attempt < max_retries:
                time.sleep(curr_backoff)
                curr_backoff *= 2.0
        except httpx.RequestError as e:
            logger.warning(f"Request error during batch extraction: {str(e)}. Retrying in {curr_backoff}s... (Attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(curr_backoff)
                curr_backoff *= 1.5

    if not response or response.status_code != 200:
        logger.error(f"Batch LLM extraction failed after {max_retries} attempts for chunks: {chunk_ids}. NO MOCK FALLBACK WILL BE USED.")
        return [], [], chunk_ids, num_429_retries

    try:
        res_json = response.json()
        if "gemini" in settings.LLM_MODEL_NAME.lower():
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raw_text = res_json["choices"][0]["message"]["content"]
            
        entities, relationships = parse_batch_response(raw_text, chunks)
        return entities, relationships, [], num_429_retries
    except Exception as e:
        logger.error(f"Failed to parse LLM batch response: {str(e)}. Marking batch as failed for chunks: {chunk_ids}.")
        return [], [], chunk_ids, num_429_retries

def build_batch_api_payload(
    chunks: List[Chunk],
    batch_size: int = 4
) -> Tuple[List[dict], Dict[str, List[Chunk]]]:
    """
    Constructs Gemini Batch API request payload items from a workload of chunks.
    Groups chunks into batches of size batch_size and maps each custom_id to its list of Chunk objects.
    """
    payload_items: List[dict] = []
    custom_id_map: Dict[str, List[Chunk]] = {}

    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i + batch_size]
        custom_id = f"batch_req_{i // batch_size}_{chunk_batch[0].chunk_id}"
        custom_id_map[custom_id] = chunk_batch

        chunks_payload = [
            {"chunk_id": c.chunk_id, "text": c.text}
            for c in chunk_batch
        ]
        
        prompt = f"""You are a clinical NLP extractor. Your task is to extract clinical entities and relationships from the provided clinical text chunks in a single structured pass.

ALLOWED ENTITY TYPES:
{', '.join(sorted(ALLOWED_ENTITY_TYPES))}

ALLOWED RELATION TYPES AND CONSTRAINTS:
- TREATS: Drug -> Condition
- CAUSES: Drug -> SideEffect
- HAS_SYMPTOM: Condition -> Symptom
- INTERACTS_WITH: Drug -> Drug
- CONTRAINDICATED_FOR: Drug -> Condition OR Drug -> Population
- RECOMMENDS: Guideline -> Drug

For each chunk, extract the entities and the relationships connecting those entities.

Return the output as a valid JSON object matching this schema:
{{
  "chunks": [
    {{
      "chunk_id": "chunk_doc1_0",
      "entities": [
        {{
          "entity_text": "metformin hcl",
          "normalized_name": "Metformin",
          "entity_type": "Drug",
          "confidence": 0.95
        }}
      ],
      "relationships": [
        {{
          "source_entity_text": "metformin hcl",
          "relation_type": "TREATS",
          "target_entity_text": "type 2 diabetes",
          "confidence": 0.90
        }}
      ]
    }}
  ]
}}

CHUNKS TO PROCESS:
{json.dumps(chunks_payload, indent=2)}
"""
        req_item = {
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
        }
        payload_items.append(req_item)

    return payload_items, custom_id_map

def submit_gemini_batch_job(
    payload_items: List[dict],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> dict:
    """
    Submits a batch request payload list to the official Google Gemini Batch API endpoint.
    Uses x-goog-api-key header for authentication and batch.input_config.requests structure.
    Returns the created batch job response JSON.
    """
    if api_key is None:
        api_key = settings.LLM_API_KEY
    if model_name is None:
        model_name = settings.LLM_MODEL_NAME

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:batchGenerateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    body = {
        "batch": {
            "display_name": "clinical-batch-extraction",
            "input_config": {
                "requests": {
                    "requests": payload_items
                }
            }
        }
    }

    logger.info(f"Submitting {len(payload_items)} request batches to Gemini Batch API endpoint ({model_name})...")
    res = httpx.post(url, json=body, headers=headers, timeout=60.0)
    res.raise_for_status()
    return res.json()

def poll_gemini_batch_job(
    job_name: str,
    api_key: Optional[str] = None,
    poll_interval: float = 10.0,
    timeout: float = 3600.0
) -> dict:
    """
    Polls the Gemini Batch API job status until completion or timeout.
    Uses x-goog-api-key header authentication against GET https://generativelanguage.googleapis.com/v1beta/{job_name}.
    """
    if api_key is None:
        api_key = settings.LLM_API_KEY

    # Standardize job resource name
    job_path = job_name.lstrip("/")
    url = f"https://generativelanguage.googleapis.com/v1beta/{job_path}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    start_t = time.time()

    while time.time() - start_t < timeout:
        res = httpx.get(url, headers=headers, timeout=30.0)
        res.raise_for_status()
        job_data = res.json()
        
        state = (
            job_data.get("state")
            or job_data.get("metadata", {}).get("state")
            or job_data.get("batch", {}).get("state", "JOB_STATE_UNSPECIFIED")
        )
        logger.info(f"Polling Gemini Batch job '{job_name}': State = {state}")

        if state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
            return job_data

        time.sleep(poll_interval)

    raise TimeoutError(f"Gemini Batch job '{job_name}' timed out after {timeout}s.")

def parse_gemini_batch_results(
    completed_job: dict,
    custom_id_map: Dict[str, List[Chunk]]
) -> Tuple[List[Entity], List[Relationship], List[str], int]:
    """
    Parses completed Gemini Batch API responses and maps results back to source chunks.
    Extracts inlined_responses, isolates failed items, and returns (all_entities, all_relationships, failed_chunk_ids, num_429).
    """
    all_entities: List[Entity] = []
    all_relationships: List[Relationship] = []
    failed_chunk_ids: List[str] = []

    # Extract response items array
    inlined = (
        completed_job.get("inlined_responses")
        or completed_job.get("destinations", [{}])[0].get("results", [])
        or completed_job.get("batch", {}).get("inlined_responses")
        or completed_job.get("batch", {}).get("destinations", [{}])[0].get("results", [])
        or []
    )

    custom_ids = list(custom_id_map.keys())

    for idx, item in enumerate(inlined):
        custom_id = item.get("custom_id") or (custom_ids[idx] if idx < len(custom_ids) else None)
        chunks = custom_id_map.get(custom_id, []) if custom_id else []
        item_chunk_ids = [c.chunk_id for c in chunks]

        status = item.get("status", {})
        if status and status.get("code", 0) != 0:
            logger.error(f"Batch API item '{custom_id}' failed with code {status.get('code')}: {status.get('message')}.")
            failed_chunk_ids.extend(item_chunk_ids)
            continue

        try:
            candidates = item.get("response", {}).get("candidates") or item.get("candidates") or []
            raw_text = candidates[0]["content"]["parts"][0]["text"]
            ents, rels = parse_batch_response(raw_text, chunks)
            all_entities.extend(ents)
            all_relationships.extend(rels)
        except Exception as e:
            logger.error(f"Failed to parse Batch API response for item '{custom_id}': {str(e)}.")
            failed_chunk_ids.extend(item_chunk_ids)

    return all_entities, all_relationships, failed_chunk_ids, 0

def extract_workload_gemini_batch_api(
    chunks: List[Chunk],
    batch_size: int = 4
) -> Tuple[List[Entity], List[Relationship], List[str], int]:
    """
    Executes a complete workload of chunks using the Gemini Batch API execution path.
    Builds payloads, submits batch job, polls for completion, and parses structured results.
    """
    if not chunks:
        return [], [], [], 0

    payload_items, custom_id_map = build_batch_api_payload(chunks, batch_size=batch_size)
    
    try:
        job_info = submit_gemini_batch_job(payload_items)
        job_name = job_info.get("name") or job_info.get("batch", {}).get("name")
        if not job_name:
            raise ValueError(f"Batch API submission response missing job 'name': {job_info}")

        completed_job = poll_gemini_batch_job(
            job_name,
            poll_interval=settings.GEMINI_BATCH_POLL_INTERVAL_SECONDS,
            timeout=settings.GEMINI_BATCH_TIMEOUT_SECONDS
        )

        state = completed_job.get("state") or completed_job.get("batch", {}).get("state")
        if state != "JOB_STATE_SUCCEEDED":
            logger.error(f"Gemini Batch job '{job_name}' did not succeed. Final state: {state}.")
            all_failed = [c.chunk_id for c in chunks]
            return [], [], all_failed, 0

        return parse_gemini_batch_results(completed_job, custom_id_map)
    except Exception as e:
        logger.error(f"Gemini Batch API execution failed: {str(e)}. Recording all workload chunks as failed.")
        all_failed = [c.chunk_id for c in chunks]
        return [], [], all_failed, 0

def extract_batch(
    chunks: List[Chunk],
    force_mock: bool = False
) -> Tuple[List[Entity], List[Relationship], List[str], int]:
    """
    Main entry point for batch extraction.
    Determines extraction mode (mock, synchronous LLM, or Batch API LLM) based on configuration.
    Returns (entities, relationships, failed_chunk_ids, num_429_retries).
    """
    if not chunks:
        return [], [], [], 0

    is_real_llm = True
    if settings.LLM_API_KEY in ["mock_key", "your_api_key_here", ""] or not settings.LLM_API_KEY:
        is_real_llm = False

    is_test_run = os.environ.get("INGESTION_TEST_RUN") == "true"

    if force_mock or not is_real_llm or is_test_run:
        logger.info(f"Extracting batch of {len(chunks)} chunks using RULE-BASED MOCK mode.")
        return extract_batch_mock(chunks)
    elif settings.GEMINI_USE_BATCH_API:
        logger.info(f"Extracting batch of {len(chunks)} chunks using GEMINI BATCH API mode.")
        return extract_workload_gemini_batch_api(chunks, batch_size=settings.LLM_BATCH_SIZE)
    else:
        logger.info(f"Extracting batch of {len(chunks)} chunks using REAL SYNCHRONOUS LLM API mode.")
        return extract_batch_llm(chunks)

