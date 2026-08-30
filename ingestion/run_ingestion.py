import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.config import settings
from backend.app.models.schemas import Chunk, Entity, Relationship
from backend.app.graph.connection import neo4j_conn
from ingestion.parsers.json_parser import parse_json_bulk
from ingestion.chunking.chunker import chunk_document
from ingestion.extraction.entity_extractor import extract_entities
from ingestion.extraction.relation_extractor import extract_relations, check_contradictions
from ingestion.extraction.batch_extractor import extract_batch
from ingestion.validation.validator import validate_extracted_data, VALIDATION_LOG_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graphrag.ingestion.runner")

def run_bulk_ingestion(file_path: str, force_mock: bool = False) -> Dict[str, Any]:
    logger.info(f"Starting bulk ingestion runner for file: {file_path}")
    
    # 1. Parse JSON
    if not os.path.exists(file_path):
        logger.error(f"Source file not found: {file_path}")
        return {
            "num_source_documents": 0,
            "num_successfully_parsed": 0,
            "num_failed_documents": 0,
            "error": "Source file not found"
        }
        
    try:
        parsed_docs = parse_json_bulk(file_path)
    except Exception as e:
        logger.error(f"Failed to parse bulk JSON: {str(e)}")
        return {
            "num_source_documents": 20, # expected
            "num_successfully_parsed": 0,
            "num_failed_documents": 20,
            "error": str(e)
        }

    num_docs = len(parsed_docs)
    logger.info(f"Successfully parsed {num_docs} documents.")
    
    # 2. Chunking
    all_chunks: List[Chunk] = []
    chunk_counter = 0
    for doc in parsed_docs:
        meta = doc["metadata"]
        text = doc["text"]
        
        # Process section-aware chunking
        chunks = chunk_document(text, meta)
        # Update chunk IDs to be unique across all docs if not already done
        for c in chunks:
            if not c.chunk_id.startswith("chunk_"):
                c.chunk_id = f"chunk_{meta['document_id']}_{chunk_counter}"
            chunk_counter += 1
        all_chunks.extend(chunks)
        
    logger.info(f"Generated {len(all_chunks)} text chunks across all documents.")

    # Check LLM extraction mode constraints
    is_real_llm = True
    if settings.LLM_API_KEY in ["mock_key", "your_api_key_here", ""] or not settings.LLM_API_KEY:
        is_real_llm = False
        
    # Check if we are running under a unit test
    is_test_run = "pytest" in sys.modules or os.environ.get("INGESTION_TEST_RUN") == "true"
    
    extraction_mode = "real LLM/API extraction" if is_real_llm and not force_mock else "mock fallback"
    logger.info(f"Configured Extraction Mode: {extraction_mode}")

    # Create directories (isolate test runs from production checkpoint data)
    if is_test_run or force_mock:
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_test")
    else:
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Save documents and chunks
    with open(os.path.join(processed_dir, "documents.json"), "w", encoding="utf-8") as f:
        json.dump([d for d in parsed_docs], f, indent=2)
        
    with open(os.path.join(processed_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([c.dict() for c in all_chunks], f, indent=2)

    # Clear validation errors log file
    if os.path.exists(VALIDATION_LOG_FILE):
        try:
            os.remove(VALIDATION_LOG_FILE)
        except Exception:
            pass

    # If NO real LLM configured and not in test run, STOP here
    if not is_real_llm and not is_test_run and not force_mock:
        logger.warning("STOPPING PIPELINE: No real LLM API Key is configured and this is not a test environment.")
        summary = {
            "extraction_mode": "stopped (mock fallback)",
            "num_source_documents": 20,
            "num_successfully_parsed": num_docs,
            "num_failed_documents": 20 - num_docs,
            "num_chunks": len(all_chunks),
            "num_entities": 0,
            "num_relationships": 0,
            "num_rejected": 0,
            "num_contradictions": 0,
            "neo4j_populated": "No (stopped before population)",
            "blockers": "Real LLM provider API key is required to execute extraction."
        }
        return summary

    # 3. Extraction Stage (Entities and Relationships via Batch Extractor)
    logger.info("Executing batch entity and relationship extraction stage...")
    import time
    start_time = time.time()
    raw_entities: List[Entity] = []
    raw_relationships: List[Relationship] = []
    
    # Try to load resumed state to prevent re-extracting already completed work
    entities_temp_path = os.path.join(processed_dir, "raw_entities_temp.json")
    relationships_temp_path = os.path.join(processed_dir, "raw_relationships_temp.json")
    
    if os.path.exists(entities_temp_path) and os.path.exists(relationships_temp_path):
        logger.info("Found incremental backup files. Loading resumed state...")
        try:
            with open(entities_temp_path, "r", encoding="utf-8") as f:
                raw_entities = [Entity(**e) for e in json.load(f)]
            with open(relationships_temp_path, "r", encoding="utf-8") as f:
                raw_relationships = [Relationship(**r) for r in json.load(f)]
            logger.info(f"Loaded {len(raw_entities)} entities and {len(raw_relationships)} relationships from backup.")
        except Exception as e:
            logger.warning(f"Could not load incremental backup files: {str(e)}. Starting fresh.")
    
    # Load list of already processed & failed chunks
    processed_chunks = []
    failed_chunks = []
    processed_chunks_path = os.path.join(processed_dir, "processed_chunks_temp.json")
    failed_chunks_path = os.path.join(processed_dir, "failed_chunks_temp.json")

    if os.path.exists(processed_chunks_path):
        try:
            with open(processed_chunks_path, "r", encoding="utf-8") as f:
                processed_chunks = json.load(f)
            logger.info(f"Resuming pipeline: {len(processed_chunks)} chunks already processed.")
        except Exception:
            pass

    if os.path.exists(failed_chunks_path):
        try:
            with open(failed_chunks_path, "r", encoding="utf-8") as f:
                failed_chunks = json.load(f)
            logger.info(f"Resuming pipeline: {len(failed_chunks)} chunks previously recorded as failed.")
        except Exception:
            pass

    previously_successful_chunks_count = len(processed_chunks)
    
    # Filter out chunks already successfully processed
    unprocessed_chunks = [c for c in all_chunks if c.chunk_id not in processed_chunks]
    # Reset failed_chunks list for this retry attempt
    failed_chunks = []
    
    batch_size = settings.LLM_BATCH_SIZE
    total_batches = (len(unprocessed_chunks) + batch_size - 1) // batch_size if unprocessed_chunks else 0
    logger.info(f"Processing {len(unprocessed_chunks)} unprocessed chunks in {total_batches} batches of size {batch_size}...")

    successful_batches = 0
    failed_batches = 0
    total_429_retries = 0

    for i in range(0, len(unprocessed_chunks), batch_size):
        batch = unprocessed_chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        entities, rels, failed_ids, num_429 = extract_batch(batch, force_mock=force_mock)
        total_429_retries += num_429

        if failed_ids:
            failed_batches += 1
        else:
            successful_batches += 1

        raw_entities.extend(entities)
        raw_relationships.extend(rels)

        # Record completed chunk IDs
        for c in batch:
            if c.chunk_id not in failed_ids:
                if c.chunk_id not in processed_chunks:
                    processed_chunks.append(c.chunk_id)
            else:
                if c.chunk_id not in failed_chunks:
                    failed_chunks.append(c.chunk_id)

        # Incremental backup to prevent data loss during long runs
        try:
            with open(os.path.join(processed_dir, "raw_entities_temp.json"), "w", encoding="utf-8") as f:
                json.dump([e.dict() for e in raw_entities], f, indent=2)
            with open(os.path.join(processed_dir, "raw_relationships_temp.json"), "w", encoding="utf-8") as f:
                json.dump([r.dict() for r in raw_relationships], f, indent=2)
            with open(processed_chunks_path, "w", encoding="utf-8") as f:
                json.dump(processed_chunks, f)
            with open(failed_chunks_path, "w", encoding="utf-8") as f:
                json.dump(failed_chunks, f)
        except Exception as e:
            logger.warning(f"Could not write incremental temp backups: {str(e)}")

        # Smooth rate limiting delay for real LLM requests
        if is_real_llm and not force_mock and not is_test_run:
            time.sleep(settings.LLM_BATCH_DELAY_SECONDS)

    total_execution_time = time.time() - start_time
    avg_time_per_batch = (total_execution_time / total_batches) if total_batches > 0 else 0.0

    logger.info(f"Extracted {len(raw_entities)} raw entities and {len(raw_relationships)} raw relationships in {total_execution_time:.2f}s.")

    # 4. Ingestion Validation and Deduplication Stage
    validated_entities, validated_relationships = validate_extracted_data(raw_entities, raw_relationships)
    
    # Save entities and relationships
    with open(os.path.join(processed_dir, "entities.json"), "w", encoding="utf-8") as f:
        json.dump([e.dict() for e in validated_entities], f, indent=2)

    with open(os.path.join(processed_dir, "relationships.json"), "w", encoding="utf-8") as f:
        json.dump([r.dict() for r in validated_relationships], f, indent=2)

    # 5. Check validation logs for rejections and contradiction counts
    num_rejected = 0
    num_contradictions = 0
    
    if os.path.exists(VALIDATION_LOG_FILE):
        with open(VALIDATION_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if "REJECTED" in line:
                    num_rejected += 1
                if "CONTRADICTION" in line or "contradiction" in line.lower():
                    num_contradictions += 1

    # 6. Neo4j Population Stage (bypassed per prompt instructions)
    neo4j_populated = "No"

    # Compute distributions
    entity_dist = {}
    for e in validated_entities:
        entity_dist[e.entity_type] = entity_dist.get(e.entity_type, 0) + 1

    rel_dist = {}
    for r in validated_relationships:
        rel_dist[r.relation_type] = rel_dist.get(r.relation_type, 0) + 1

    file_sizes = {
        "documents.json": os.path.getsize(os.path.join(processed_dir, "documents.json")),
        "chunks.json": os.path.getsize(os.path.join(processed_dir, "chunks.json")),
        "entities.json": os.path.getsize(os.path.join(processed_dir, "entities.json")),
        "relationships.json": os.path.getsize(os.path.join(processed_dir, "relationships.json"))
    }

    newly_successful_chunks_count = len(processed_chunks) - previously_successful_chunks_count
    remaining_failed_chunks_count = len(failed_chunks)

    summary = {
        "extraction_mode": extraction_mode,
        "num_source_documents": 20,
        "num_successfully_parsed": num_docs,
        "num_failed_documents": 20 - num_docs,
        "num_chunks": len(all_chunks),
        "num_failed_chunks": remaining_failed_chunks_count,
        "num_entities": len(validated_entities),
        "num_relationships": len(validated_relationships),
        "num_rejected": num_rejected,
        "num_contradictions": num_contradictions,
        "total_chunks": len(all_chunks),
        "previously_successful_chunks": previously_successful_chunks_count,
        "newly_successful_chunks": newly_successful_chunks_count,
        "remaining_failed_chunks": remaining_failed_chunks_count,
        "documents_processed": num_docs,
        "chunks_processed": len(all_chunks),
        "batches_processed": total_batches,
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "raw_entity_count": len(raw_entities),
        "unique_validated_entity_count": len(validated_entities),
        "entity_distribution_by_type": entity_dist,
        "raw_relationship_count": len(raw_relationships),
        "unique_validated_relationship_count": len(validated_relationships),
        "relationship_distribution_by_type": rel_dist,
        "validation_rejections": num_rejected,
        "contradiction_flags": num_contradictions,
        "extraction_failures": remaining_failed_chunks_count,
        "http_429_occurrences": total_429_retries,
        "total_execution_time_seconds": round(total_execution_time, 2),
        "average_time_per_batch_seconds": round(avg_time_per_batch, 2),
        "final_output_file_sizes_bytes": file_sizes,
        "neo4j_populated": neo4j_populated,
        "output_files": [
            "data/processed/documents.json",
            "data/processed/chunks.json",
            "data/processed/entities.json",
            "data/processed/relationships.json",
            "data/processed/ingestion_summary.json"
        ]
    }

    # Save machine-readable ingestion summary
    summary_file = os.path.join(processed_dir, "ingestion_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    file_sizes["ingestion_summary.json"] = os.path.getsize(summary_file)
    summary["final_output_file_sizes_bytes"] = file_sizes

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Ingestion Pipeline Runner")
    parser.add_argument(
        "--file", 
        default=r"data/raw/openfda/openfda_labels_20.json",
        help="Path to the openFDA JSON labels dataset file."
    )
    parser.add_argument(
        "--force-mock", 
        action="store_true",
        help="Bypasses LLM stop check to complete pipeline run using mock fallback."
    )
    args = parser.parse_args()
    
    summary = run_bulk_ingestion(args.file, force_mock=args.force_mock)
    
    print("\n" + "="*40)
    print("        INGESTION RUNNER SUMMARY")
    print("="*40)
    for k, v in summary.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
    print("="*40 + "\n")
