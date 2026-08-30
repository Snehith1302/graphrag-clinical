import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from ingestion.parsers.cleaner import clean_text

logger = logging.getLogger("graphrag.ingestion.json_parser")

def _parse_single_record(label_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parses a single openFDA drug label record dictionary.
    Returns:
        Dict[str, Any]: A document dictionary with "text" and "metadata" keys,
                        or None if the record is invalid.
    """
    if not label_data or not isinstance(label_data, dict):
        logger.warning("Empty or malformed record encountered in openFDA JSON.")
        return None

    # Extract openfda metadata block
    openfda = label_data.get("openfda", {})
    if not isinstance(openfda, dict):
        openfda = {}
    
    # 1. Title/Name
    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    title = ""
    if brand_names and isinstance(brand_names, list):
        title = brand_names[0]
    elif generic_names and isinstance(generic_names, list):
        title = generic_names[0]
    else:
        title = label_data.get("title", "Unknown Drug Label")
 
    # 2. Document ID (Prefer stable 'id' field)
    doc_id = label_data.get("id")
    if not doc_id:
        doc_id = title.lower().replace(" ", "_")

    # 3. Publisher / Manufacturer
    manufacturers = openfda.get("manufacturer_name", [])
    publisher = "FDA"
    if manufacturers and isinstance(manufacturers, list):
        publisher = manufacturers[0]

    # 4. Year from effective_time
    effective_time = label_data.get("effective_time")
    year = datetime.now().year
    if effective_time and isinstance(effective_time, str) and len(effective_time) >= 4:
        try:
            year = int(effective_time[:4])
        except ValueError:
            pass

    # 5. Extract all key clinical sections
    sections = {
        "Indications": ["indications_and_usage", "indications"],
        "Contraindications": ["contraindications", "contraindication"],
        "Adverse Reactions": ["adverse_reactions", "adverse_reaction", "side_effects"],
        "Warnings and Precautions": ["warnings_and_precautions", "warnings"],
        "Drug Interactions": ["drug_interactions", "interactions"],
        "Use in Specific Populations": ["use_in_specific_populations"],
        "Clinical Pharmacology": ["clinical_pharmacology"],
        "Mechanism of Action": ["mechanism_of_action"]
    }

    text_parts = []
    for heading, keys in sections.items():
        content_list = []
        for key in keys:
            val = label_data.get(key)
            if val:
                if isinstance(val, list):
                    content_list.extend(val)
                else:
                    content_list.append(str(val))
        
        if content_list:
            joined = "\n".join(content_list)
            cleaned = clean_text(joined)
            if cleaned:
                text_parts.append(f"# {heading}\n{cleaned}")

    full_text = "\n\n".join(text_parts)
    
    # Exclude entirely empty documents
    if not full_text.strip():
        logger.warning(f"Skipping record ID '{doc_id}' because it has no clinical section content.")
        return None

    # Resolve source URL
    url = None
    url_val = openfda.get("url")
    if isinstance(url_val, list) and url_val:
        url = url_val[0]
    elif isinstance(url_val, str):
        url = url_val

    metadata = {
        "document_id": doc_id,
        "title": title,
        "source_type": "label",
        "publisher": publisher,
        "authors": [],  # Labels do not have academic authors
        "year": year,
        "url": url,
        "ingestion_date": datetime.utcnow().isoformat()
    }

    return {
        "text": full_text,
        "metadata": metadata
    }

def parse_json(file_path: str) -> Dict[str, Any]:
    """
    Parses the first record of an openFDA-style JSON file (for backward compatibility).
    """
    logger.info(f"Parsing JSON label file (single): {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        label_data = data["results"][0] if len(data["results"]) > 0 else {}
    elif isinstance(data, dict):
        label_data = data
    else:
        raise ValueError("Invalid JSON format: expected a dictionary.")

    doc = _parse_single_record(label_data)
    if not doc:
        raise ValueError("JSON file contains no valid clinical label records.")
    return doc

def parse_json_bulk(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses ALL records in the results array of the openFDA JSON file.
    Skips invalid records with warnings instead of crashing.
    """
    logger.info(f"Parsing JSON label file (bulk): {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        records = data["results"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        raise ValueError("Invalid JSON format: expected a dictionary or list.")

    parsed_documents = []
    for r in records:
        try:
            doc = _parse_single_record(r)
            if doc:
                parsed_documents.append(doc)
        except Exception as e:
            logger.error(f"Error parsing drug label record: {str(e)}")
            continue

    logger.info(f"Bulk parsed {len(parsed_documents)} valid documents from {file_path}.")
    return parsed_documents
