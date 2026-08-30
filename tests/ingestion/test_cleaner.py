"""
Unit tests for text cleaning and quality validation.
"""
from ingestion.parsers.cleaner import clean_text, validate_text

def test_clean_text_whitespace():
    # Constructing a string containing extra whitespace, tabs, and newlines
    text = "This   is \t a   text\n\n\n\nwith too   many newlines  and spaces."
    cleaned = clean_text(text)
    
    assert "  " not in cleaned  # no multiple spaces
    assert "\t" not in cleaned  # no tabs
    assert "\n\n\n" not in cleaned  # no more than double newlines
    assert cleaned.startswith("This is")

def test_clean_text_page_numbers():
    text = "Section 1. Metformin Guidelines. Page 5 of 12\nThis is the content on page 5.\nPage 6"
    cleaned = clean_text(text)
    assert "Page 5 of 12" not in cleaned
    assert "Page 6" not in cleaned
    assert "Section 1. Metformin Guidelines." in cleaned

def test_clean_text_encoding():
    # Text with smart quotes (\u201c, \u201d) and em-dash (\u2014)
    text = "Metformin \u201csmart quotes\u201d and \u2014 dashes."
    cleaned = clean_text(text)
    assert '"smart quotes"' in cleaned
    assert '-' in cleaned

def test_validate_text_threshold():
    short_text = "Short clinical text under hundred."
    long_text = "Metformin Hydrochloride is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus. It should be taken with meals."
    assert validate_text(short_text) is False
    assert validate_text(long_text) is True
