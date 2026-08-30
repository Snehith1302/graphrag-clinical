import re
import logging

logger = logging.getLogger("graphrag.ingestion.cleaner")

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from documents:
    - Normalizes white spaces (tabs, multiple spaces, multiple newlines).
    - Removes common page number patterns (e.g. "Page 1 of 10", "Page 5").
    - Removes typical PDF artifacts.
    - Fixes encoding abnormalities (e.g., smart quotes, accents, bad characters).
    """
    if not text:
        return ""

    # Fix common encoding anomalies / smart punctuation
    replacements = {
        "\u201c": '"', "\u201d": '"',  # double quotes
        "\u2018": "'", "\u2019": "'",  # single quotes
        "\u2013": "-", "\u2014": "-",  # dashes
        "\u2022": "*",                 # bullets
        "\u00a0": " ",                 # non-breaking space
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)

    # Remove page numbers like "Page X", "Page X of Y", "Page: X", or footer page numbers
    text = re.sub(r"(?i)\bpage\s+\d+\s+of\s+\d+\b", "", text)
    text = re.sub(r"(?i)\bpage\s+\d+\b", "", text)

    # Normalize whitespace: replace tabs and multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Normalize multiple newlines to a maximum of two newlines (to preserve paragraph structure)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    
    # Clean leading and trailing whitespace
    text = text.strip()

    return text

def validate_text(text: str) -> bool:
    """
    Validates text quality:
    - Returns False if length of text is under 100 characters (Section 8: Flag documents with <100 chars post-clean as failed).
    """
    cleaned = clean_text(text)
    length = len(cleaned)
    if length < 100:
        logger.warning(f"Document validation failed: cleaned text length ({length}) is under 100 characters.")
        return False
    return True
