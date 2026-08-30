"""
Section-aware and sliding-window document chunking module.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from backend.app.models.schemas import Chunk

logger = logging.getLogger("graphrag.ingestion.chunker")

def split_by_sections(text: str) -> List[Dict[str, Any]]:
    """
    Splits text by Markdown H1 headings ('# Heading Name').
    Returns a list of dictionaries, each with:
    - "title": Header title string (or None for pre-heading text).
    - "text": The text content of the section.
    - "start_char": Character index start.
    - "end_char": Character index end.
    """
    # Find all matches of headings like "^# (.+)$" at start of lines
    matches = list(re.finditer(r'^#\s+(.+)$', text, re.MULTILINE))
    
    sections = []
    if not matches:
        # No sections found, treat the entire document as a single section
        return [{"title": None, "text": text, "start_char": 0, "end_char": len(text)}]
        
    # Extract any text before the first section heading (e.g. document introduction)
    first_match = matches[0]
    if first_match.start() > 0:
        pre_text = text[:first_match.start()]
        if pre_text.strip():
            sections.append({
                "title": None,
                "text": pre_text,
                "start_char": 0,
                "end_char": first_match.start()
            })
            
    # Extract each section with its heading title
    for i, current_match in enumerate(matches):
        title = current_match.group(1).strip()
        start_char = current_match.start()
        
        if i + 1 < len(matches):
            end_char = matches[i + 1].start()
        else:
            end_char = len(text)
            
        sec_text = text[start_char:end_char]
        sections.append({
            "title": title,
            "text": sec_text,
            "start_char": start_char,
            "end_char": end_char
        })
        
    return sections

def chunk_document(document_text: str, doc_metadata: Dict[str, Any], chunk_size: int = 700, overlap: int = 100) -> List[Chunk]:
    """
    Chunks document text into section-aware segments, with a sliding-window fallback.
    - Tokenization is whitespace/word-based.
    - Respects section boundaries (no overlap across different sections to preserve context).
    - Returns a list of Pydantic Chunk objects.
    """
    if not document_text.strip():
        logger.warning("Empty document text passed to chunk_document.")
        return []
        
    document_id = doc_metadata.get("document_id", "unknown_doc")
    
    # Validate overlap constraint to prevent infinite loops
    if overlap >= chunk_size:
        logger.warning(f"Overlap ({overlap}) is >= chunk_size ({chunk_size}). Resetting overlap to 15% of chunk_size.")
        overlap = int(chunk_size * 0.15)
        
    # 1. Tokenize entire document into word spans with offsets
    # Each token is a tuple: (word, start_char_index, end_char_index)
    tokens = []
    for match in re.finditer(r'\S+', document_text):
        tokens.append((match.group(0), match.start(), match.end()))
        
    if not tokens:
        logger.warning("No tokens found in document text.")
        return []
        
    # 2. Split document into sections
    sections = split_by_sections(document_text)
    
    chunks = []
    chunk_index = 0
    
    # 3. Process each section independently
    for section in sections:
        sec_title = section["title"]
        start_char = section["start_char"]
        end_char = section["end_char"]
        
        # Filter global tokens that fall inside this section's character boundaries
        sec_tokens = [t for t in tokens if t[1] >= start_char and t[2] <= end_char]
        
        if not sec_tokens:
            continue
            
        # Apply sliding window chunking to this section's tokens
        start_idx = 0
        while start_idx < len(sec_tokens):
            chunk_tokens = sec_tokens[start_idx : start_idx + chunk_size]
            if not chunk_tokens:
                break
                
            # Get global character boundaries for this slice of tokens
            chunk_start = chunk_tokens[0][1]
            chunk_end = chunk_tokens[-1][2]
            chunk_text = document_text[chunk_start:chunk_end]
            
            # Deterministic ID generation: {doc_id}_chunk_{index}
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                start_offset=chunk_start,
                end_offset=chunk_end,
                section_title=sec_title,
                embedding_id=None  # Placeholder for vector index IDs
            ))
            
            chunk_index += 1
            
            # Exit loop if all tokens were processed
            if len(sec_tokens) <= chunk_size:
                break
                
            # Advance start index
            start_idx += (chunk_size - overlap)
            
    logger.info(f"Generated {len(chunks)} chunks for document: {document_id}")
    return chunks
