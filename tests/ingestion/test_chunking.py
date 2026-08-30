"""
Unit tests for the section-aware and sliding-window document chunker.
"""
from ingestion.chunking.chunker import chunk_document

def test_chunker_short_document():
    # Test doc smaller than chunk_size (10 words)
    doc_text = "This is a short clinical document about Metformin indications."
    metadata = {"document_id": "short_doc"}
    chunks = chunk_document(doc_text, metadata, chunk_size=10, overlap=2)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "short_doc"
    assert chunk.chunk_id == "short_doc_chunk_0"
    assert chunk.text == doc_text
    assert chunk.start_offset == 0
    assert chunk.end_offset == len(doc_text)
    assert chunk.section_title is None

def test_chunker_sliding_window():
    # Document with 25 words, chunk_size=10, overlap=3
    # Chunk step = 10 - 3 = 7
    # Chunks expected:
    # Chunk 0: index 0 to 9 (10 words)
    # Chunk 1: index 7 to 16 (10 words)
    # Chunk 2: index 14 to 23 (10 words)
    # Chunk 3: index 21 to 24 (4 words)
    words = [f"word{i}" for i in range(25)]
    doc_text = " ".join(words)
    metadata = {"document_id": "sliding_doc"}
    
    chunks = chunk_document(doc_text, metadata, chunk_size=10, overlap=3)
    assert len(chunks) == 4
    
    # Verify offsets correspond to actual text slice
    for chunk in chunks:
        assert doc_text[chunk.start_offset:chunk.end_offset] == chunk.text
        
    # Verify overlap behavior: check that last 3 words of chunk 0 match first 3 words of chunk 1
    c0_words = chunks[0].text.split()
    c1_words = chunks[1].text.split()
    assert c0_words[-3:] == c1_words[:3]

def test_chunker_section_aware():
    # Document with Markdown H1 sections
    doc_text = (
        "Document Introduction. This describes general document structure.\n\n"
        "# Section One\n"
        "Metformin treatment guidelines first line therapy.\n\n"
        "# Section Two\n"
        "Contraindications severe renal impairment."
    )
    
    metadata = {"document_id": "section_doc"}
    chunks = chunk_document(doc_text, metadata, chunk_size=50, overlap=5)
    
    # Expecting 3 chunks corresponding to:
    # 1. Introduction (pre-section)
    # 2. Section One
    # 3. Section Two
    assert len(chunks) == 3
    
    assert chunks[0].section_title is None
    assert chunks[1].section_title == "Section One"
    assert chunks[2].section_title == "Section Two"
    
    # Confirm offsets are globally correct
    for chunk in chunks:
        assert doc_text[chunk.start_offset:chunk.end_offset] == chunk.text
        assert chunk.document_id == "section_doc"

def test_chunker_section_aware_sliding():
    # A single section exceeding chunk size
    doc_text = (
        "# Main Section\n" +
        " ".join([f"word{i}" for i in range(20)])
    )
    
    metadata = {"document_id": "big_section_doc"}
    # chunk_size=10, overlap=2
    chunks = chunk_document(doc_text, metadata, chunk_size=10, overlap=2)
    
    # The title `# Main Section` counts as 3 tokens ("#", "Main", "Section")
    # Total tokens = 23. Step = 10 - 2 = 8
    # Chunk 0: 0 to 9 (10 words)
    # Chunk 1: 8 to 17 (10 words)
    # Chunk 2: 16 to 22 (7 words)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.section_title == "Main Section"
        assert doc_text[chunk.start_offset:chunk.end_offset] == chunk.text

def test_chunker_empty_edge():
    metadata = {"document_id": "empty_doc"}
    assert chunk_document("", metadata) == []
    assert chunk_document("   \n  ", metadata) == []

def test_chunker_deterministic_ids():
    doc_text = "# Section\nThis is a sample document for testing ID stability."
    metadata = {"document_id": "stable_doc"}
    
    chunks_run1 = chunk_document(doc_text, metadata)
    chunks_run2 = chunk_document(doc_text, metadata)
    
    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.start_offset == c2.start_offset
        assert c1.end_offset == c2.end_offset
