import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List
from ingestion.parsers.cleaner import clean_text

logger = logging.getLogger("graphrag.ingestion.xml_parser")

def parse_xml(file_path: str) -> Dict[str, Any]:
    """
    Parses PubMed-style XML clinical literature articles.
    Returns a dictionary with keys:
    - "text": Cleaned text of abstract/body.
    - "metadata": Dict containing document metadata.
    """
    logger.info(f"Parsing XML PubMed file: {file_path}")

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML file {file_path}: {str(e)}")
        raise e

    # Find the citation block (handling potential namespace prefixes if present)
    # PubMed XML typically starts with PubmedArticle Set or directly PubmedArticle
    article_node = root.find(".//PubmedArticle")
    if article_node is None:
        article_node = root  # Fallback to root if directly at article level

    # 1. Document ID (PMID)
    pmid_node = article_node.find(".//PMID")
    doc_id = pmid_node.text.strip() if pmid_node is not None and pmid_node.text else "unknown_pmid"

    # 2. Title (ArticleTitle)
    title_node = article_node.find(".//ArticleTitle")
    title = title_node.text.strip() if title_node is not None and title_node.text else "Unknown Article Title"
    # Remove final dot if present in title
    if title.endswith("."):
        title = title[:-1]

    # 3. Publisher (Journal Title)
    journal_node = article_node.find(".//Journal/Title")
    publisher = journal_node.text.strip() if journal_node is not None and journal_node.text else "Unknown Journal"

    # 4. Year
    year = datetime.now().year
    year_node = article_node.find(".//JournalIssue/PubDate/Year")
    if year_node is not None and year_node.text:
        try:
            year = int(year_node.text.strip()[:4])
        except ValueError:
            pass
    else:
        # Try finding MedlineDate containing a year (e.g. "2020 Jan-Feb")
        medline_date_node = article_node.find(".//JournalIssue/PubDate/MedlineDate")
        if medline_date_node is not None and medline_date_node.text:
            import re
            match = re.search(r"\b(19|20)\d{2}\b", medline_date_node.text)
            if match:
                year = int(match.group(0))

    # 5. Authors list
    authors = []
    author_nodes = article_node.findall(".//AuthorList/Author")
    for author in author_nodes:
        last_name_node = author.find("LastName")
        fore_name_node = author.find("ForeName")
        last_name = last_name_node.text.strip() if last_name_node is not None and last_name_node.text else ""
        fore_name = fore_name_node.text.strip() if fore_name_node is not None and fore_name_node.text else ""
        if last_name and fore_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name:
            authors.append(last_name)

    # 6. Extract abstract text (supporting multiple abstract texts like background, methods, results, etc.)
    abstract_parts = []
    abstract_texts = article_node.findall(".//Abstract/AbstractText")
    for abs_text in abstract_texts:
        label = abs_text.attrib.get("Label")
        text_val = "".join(abs_text.itertext()).strip()  # get all nested text
        if text_val:
            if label:
                abstract_parts.append(f"# {label.capitalize()}\n{text_val}")
            else:
                abstract_parts.append(text_val)

    # If no abstract, check for body text
    body_text_node = article_node.find(".//Body")
    if not abstract_parts and body_text_node is not None:
        abstract_parts.append("".join(body_text_node.itertext()).strip())

    full_text = "\n\n".join(abstract_parts)
    
    # Prepend Title to the text content to provide context
    if title:
        full_text = f"# {title}\n\n{full_text}"

    # Clean text content
    full_text = clean_text(full_text)

    metadata = {
        "document_id": doc_id,
        "title": title,
        "source_type": "study",
        "publisher": publisher,
        "authors": authors,
        "year": year,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/" if doc_id != "unknown_pmid" else None,
        "ingestion_date": datetime.utcnow().isoformat()
    }

    return {
        "text": full_text,
        "metadata": metadata
    }
