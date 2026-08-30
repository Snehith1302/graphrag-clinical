"""
Unit tests for JSON, XML, and PDF document parsers.
"""
import os
from datetime import datetime
from ingestion.parsers.json_parser import parse_json, parse_json_bulk
from ingestion.parsers.xml_parser import parse_xml
from ingestion.parsers.pdf_parser import parse_pdf

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "openfda")

def test_parse_json():
    json_path = os.path.join(FIXTURES_DIR, "sample.json")
    result = parse_json(json_path)
    
    assert "text" in result
    assert "metadata" in result
    
    meta = result["metadata"]
    assert meta["document_id"] == "label_metformin_001"
    assert meta["title"] == "Metformin"
    assert meta["source_type"] == "label"
    assert meta["publisher"] == "Generic Pharma Corp"
    assert meta["year"] == 2024
    assert meta["url"] == "https://open.fda.gov/drugs/label/metformin"
    
    text = result["text"]
    assert "# Indications" in text
    assert "# Contraindications" in text
    assert "severe renal impairment" in text.lower()

def test_parse_xml():
    xml_path = os.path.join(FIXTURES_DIR, "sample.xml")
    result = parse_xml(xml_path)
    
    assert "text" in result
    assert "metadata" in result
    
    meta = result["metadata"]
    assert meta["document_id"] == "32895678"
    assert meta["title"] == "Efficacy and Side Effects of Metformin in Type 2 Diabetes"
    assert meta["source_type"] == "study"
    assert meta["publisher"] == "Journal of Clinical Endocrinology"
    assert meta["year"] == 2022
    assert "Alice R Smith" in meta["authors"]
    assert "Robert B Jones" in meta["authors"]
    assert meta["url"] == "https://pubmed.ncbi.nlm.nih.gov/32895678/"
    
    text = result["text"]
    assert "# Background" in text
    assert "# Results" in text
    assert "efficacy" in text.lower()

def test_parse_pdf():
    pdf_path = os.path.join(FIXTURES_DIR, "sample.pdf")
    result = parse_pdf(pdf_path)
    
    assert "text" in result
    assert "metadata" in result
    
    meta = result["metadata"]
    assert meta["document_id"] == "sample"
    assert "Metformin Clinical Guideline" in meta["title"]
    assert meta["source_type"] == "guideline"
    assert meta["publisher"] == "WHO Publisher"
    # Year will match the PDF creation year (current year)
    assert meta["year"] == datetime.now().year
    assert "Alice Smith" in meta["authors"]
    assert "Robert Jones" in meta["authors"]
    assert meta["url"] == "https://www.who.int/publications/metformin-guideline"
    
    text = result["text"]
    assert "indications" in text.lower()
    assert "severe renal impairment" in text.lower()
    assert "lactic acidosis" in text.lower()

def test_parse_json_bulk():
    json_path = os.path.join(FIXTURES_DIR, "sample.json")
    results = parse_json_bulk(json_path)
    
    assert isinstance(results, list)
    assert len(results) == 1
    
    doc = results[0]
    assert "text" in doc
    assert "metadata" in doc
    assert doc["metadata"]["document_id"] == "label_metformin_001"

def test_parse_json_bulk_real_dataset():
    real_path = os.path.join(REAL_DATA_DIR, "openfda_labels_20.json")
    if not os.path.exists(real_path):
        return  # Skip if real dataset is not present in local test run environment
        
    results = parse_json_bulk(real_path)
    
    assert isinstance(results, list)
    assert len(results) == 20
    
    doc_ids = [d["metadata"]["document_id"] for d in results]
    assert len(set(doc_ids)) == 20  # All unique
    
    for doc in results:
        meta = doc["metadata"]
        assert meta["source_type"] == "label"
        assert meta["publisher"] != ""
        assert "document_id" in meta
        assert "text" in doc
        
        # Check text structure
        text = doc["text"]
        assert len(text.strip()) > 0
        # Check section heading presence
        assert "# Indications" in text or "Clinical Pharmacology" in text or "Warnings" in text

