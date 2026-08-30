import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import pdfplumber
from ingestion.parsers.cleaner import clean_text

logger = logging.getLogger("graphrag.ingestion.pdf_parser")

def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Parses PDF clinical guideline documents using pdfplumber.
    Returns a dictionary with keys:
    - "text": Cleaned concatenated text of all pages.
    - "metadata": Dict containing document metadata.
    """
    logger.info(f"Parsing PDF file: {file_path}")

    filename = os.path.basename(file_path)
    file_base, _ = os.path.splitext(filename)

    # 1. Extract text page-by-page
    page_texts = []
    metadata_extracted = {}
    
    try:
        with pdfplumber.open(file_path) as pdf:
            metadata_extracted = pdf.metadata or {}
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Clean page number footer patterns directly if we can, 
                    # but general cleaning will happen on the full text
                    page_texts.append(text)
                else:
                    logger.warning(f"Page {i+1} in PDF {file_path} returned empty text.")
    except Exception as e:
        logger.error(f"Failed to open/parse PDF file {file_path}: {str(e)}")
        raise e

    full_text = "\n\n".join(page_texts)
    full_text = clean_text(full_text)

    # 2. Extract metadata
    # Document ID: slugified filename
    doc_id = file_base.lower().replace(" ", "_").replace("-", "_")

    # Title: PDF title property or file base name
    title = metadata_extracted.get("Title")
    if not title or title.strip() == "" or len(title.strip()) < 3:
        # Replace underscores with spaces for readability as a fallback title
        title = file_base.replace("_", " ").replace("-", " ").title()

    # Publisher: Creator/Producer/Author property or "Unknown Guideline Publisher"
    publisher = metadata_extracted.get("Creator")
    if not publisher:
        publisher = metadata_extracted.get("Producer")
    if not publisher:
        publisher = "Unknown Guideline Publisher"

    # Authors list: split Author metadata field if it exists
    authors = []
    author_str = metadata_extracted.get("Author")
    if author_str:
        # Split by comma or semicolon
        for part in author_str.replace(";", ",").split(","):
            cleaned_author = part.strip()
            if cleaned_author:
                authors.append(cleaned_author)

    # Year: parse from CreationDate (usually D:YYYYMMDDHHMMSS...)
    year = datetime.now().year
    creation_date = metadata_extracted.get("CreationDate")
    if creation_date and len(creation_date) >= 6:
        # CreationDate is often formatted like "D:20241012112233..."
        clean_date = creation_date.replace("D:", "")
        if len(clean_date) >= 4:
            try:
                year = int(clean_date[:4])
            except ValueError:
                pass

    metadata = {
        "document_id": doc_id,
        "title": title,
        "source_type": "guideline",
        "publisher": publisher,
        "authors": authors,
        "year": year,
        "url": metadata_extracted.get("Subject") if metadata_extracted.get("Subject", "").startswith("http") else None,
        "ingestion_date": datetime.utcnow().isoformat()
    }

    return {
        "text": full_text,
        "metadata": metadata
    }
