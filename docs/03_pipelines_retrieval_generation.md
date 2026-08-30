# 03 — Ingestion, Extraction, Retrieval, and Generation Pipelines

## SECTION 8 — DOCUMENT INGESTION PIPELINE

`Document → parsing → cleaning → normalization → metadata extraction → chunking → entity extraction → relation extraction → validation → graph insertion → vector indexing`

| Stage | Input | Output | Purpose | Recommended library | Failure handling |
|---|---|---|---|---|---|
| Parsing | Raw PDF/JSON/XML | Plain text + structure markers (headings/sections) | Extract readable text | `pdfplumber`/`PyMuPDF` (PDF), `xml.etree`/`lxml` (PubMed XML), `json` (openFDA) | Log + skip file, do not crash pipeline |
| Cleaning | Raw text | Normalized text (whitespace, encoding fixed) | Remove artifacts, headers/footers, page numbers | `re`, custom rules | Flag documents with <100 chars post-clean as failed |
| Normalization | Cleaned text | Consistent casing/terminology where feasible | Reduce entity duplication downstream | Custom regex + a small synonym map | Non-blocking — normalization improves quality, isn't required to proceed |
| Metadata extraction | Document + filename/source | `Document` metadata object | Provenance backbone | Custom parser per source type | Required fields missing → mark `incomplete_metadata=true`, still ingest but flag in UI |
| Chunking | Normalized text | List of chunks with offsets | Prepare for embeddings + extraction context window | Custom (section-aware) chunker, ~500–800 tokens, 15% overlap | Chunk exceeding model context → split further |
| Entity extraction | Chunk text | List of `Entity` objects | Populate candidate nodes | spaCy + LLM-assisted (see Section 9) | Extraction failure on a chunk → log, continue with next chunk (don't fail whole doc) |
| Relation extraction | Chunk text + extracted entities | List of `Relation` objects | Populate candidate edges | LLM-assisted structured extraction (see Section 10) | Same as above — per-chunk isolation |
| Validation | Candidate entities/relations | Validated entities/relations | Schema conformance, dedup, confidence threshold | Custom validator against JSON schema | Reject non-conforming records, log to `validation_errors.log` |
| Graph insertion | Validated entities/relations | Neo4j nodes/edges | Persist knowledge graph | `neo4j` Python driver, `MERGE` (idempotent) | Insertion failure → retry once, then log and skip record |
| Vector indexing | Chunks | Embedded vectors in vector store | Enable vector RAG baseline | `sentence-transformers` + FAISS/Qdrant/pgvector | Embedding failure → skip chunk, log |

**Chunking strategy:** section-aware where structure is available (guideline headings), fallback to sliding window (700 tokens, 100 token overlap) otherwise. Store `chunk_id`, `document_id`, `start_offset`, `end_offset`, `section_title` (if known).

**Metadata schema (per document):** `document_id`, `title`, `source_type` (guideline/study/label), `publisher/journal`, `authors[]`, `year`, `url` (if reusable), `ingestion_date`, `license_note`.

---

## SECTION 9 — ENTITY EXTRACTION

| Approach | Pros | Cons |
|---|---|---|
| spaCy (generic NER) | Fast, free, no API cost | Poor recall on clinical entity types out of the box |
| Biomedical NER model (e.g., scispaCy) | Purpose-built for biomedical text, decent recall | Still needs mapping to your custom schema types |
| Pure LLM extraction | Flexible, can extract directly into your schema, handles paraphrasing well | API cost, needs strict prompting + validation, non-deterministic |
| Hybrid (scispaCy candidate detection + LLM structuring/typing) | Best precision/recall balance for a bounded project | More pipeline complexity |

**Recommended architecture:** scispaCy (or spaCy `en_core_web_sm` if scispaCy setup is too heavy for the timeline) for candidate span detection → LLM call to classify each candidate into one of the 7 schema node types and normalize its name, with structured JSON output. This bounds LLM cost (only classifying pre-detected spans, not re-reading full documents) while keeping schema-conformance high.

**Entity JSON schema:**
```json
{
  "entity_text": "metformin",
  "normalized_name": "Metformin",
  "entity_type": "Drug",
  "confidence": 0.93,
  "document_id": "label_003",
  "section": "Indications",
  "source_span": [1204, 1213],
  "extraction_method": "hybrid_scispacy_llm"
}
```

**Normalization & deduplication:** lowercase + strip punctuation for a normalization key; maintain a small synonym/alias map (e.g., "Type 2 Diabetes Mellitus" → "Type 2 Diabetes"); on insertion, `MERGE` by `normalized_name` + `entity_type` so duplicate mentions collapse to one node with multiple provenance links.

---

## SECTION 10 — RELATION EXTRACTION

**Method:** LLM-based structured extraction, given a chunk and its already-extracted entities, prompted to output only relations from the **allowed relation type list** (Section 6) with confidence and source span — never free-text relation types.

**Allowed relation types:** `TREATS`, `CAUSES`, `HAS_SYMPTOM`, `INTERACTS_WITH`, `CONTRAINDICATED_FOR`, `RECOMMENDS` (schema is closed-set by design — this is what makes downstream Cypher generation safe, see Section 14).

**Relation JSON schema:**
```json
{
  "source_entity": "Metformin",
  "source_type": "Drug",
  "relation_type": "CONTRAINDICATED_FOR",
  "target_entity": "Severe Renal Impairment",
  "target_type": "Population",
  "confidence": 0.88,
  "document_id": "label_003",
  "source_span": [2210, 2340],
  "extraction_method": "llm_structured"
}
```

**Schema validation:** reject any relation whose `relation_type` is not in the allowed list, or whose `source_type`/`target_type` pair doesn't match Section 6's defined source→target mapping.

**Confidence scoring:** LLM self-reported confidence (0–1) + a rule-based boost/penalty (e.g., +0.05 if both entities also co-occur in a second independent document, −0.1 if extracted from a low-confidence entity pair).

**Duplicate handling:** on `MERGE`, if an identical (source, relation_type, target) triple already exists from a different document, keep both provenance links on the same edge (`source_ids: []` array) rather than creating duplicate edges.

**Contradictory relationship handling:** if two documents assert conflicting relations (e.g., one says `CONTRAINDICATED_FOR`, another implies safety), **do not silently pick one** — store both with their sources and surface the conflict explicitly in the UI evidence panel ("Sources disagree — see both"). This is safer and more honest than resolving silently.

**LLM-generated relation validation before Neo4j insertion:** (1) schema/type check, (2) confidence threshold (default ≥0.6, configurable), (3) entity existence check (both entities must already exist or be simultaneously created in the same transaction), (4) provenance required (no `source_id` → reject).

---

## SECTION 11 — VECTOR RAG BASELINE

`Question → embedding → vector search → top-K retrieval → context construction → LLM → answer`

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (fast, free, sufficient for a bounded corpus) or a stronger biomedical embedding model (e.g., a PubMedBERT-based sentence embedder) if time allows — recommend starting with MiniLM for speed, upgrading only if baseline quality is clearly insufficient.
- **Chunk size:** 700 tokens, **overlap:** 100 tokens (matches ingestion chunking in Section 8, so both pipelines share identical chunks — necessary for a fair comparison)
- **Top-K:** 5 (configurable), **similarity measure:** cosine similarity
- **Metadata filtering:** optional filter by `source_type` if the query implies one (e.g., "according to the guideline")
- **Context construction:** concatenate top-K chunks with clear `[Source: doc_title, section]` markers before each chunk
- **Prompt structure:** see Section 15's shared generation prompt — vector RAG and GraphRAG use the *same* generation prompt template, differing only in what evidence is injected, to keep the comparison fair
- **Citation handling:** every chunk injected into context carries its `document_id`; the LLM is instructed to cite `document_id` inline, which the backend resolves to full citation metadata for display

This baseline must be a **fair, competent baseline** — not a strawman — since the whole research question depends on comparing against a real system, not a weak one.

---

## SECTION 12 — GRAPH RAG PIPELINE

`Question → query understanding → entity identification → graph node matching → graph traversal → multi-hop retrieval → evidence collection → optional vector retrieval → evidence fusion → LLM generation → citations`

1. **Query understanding:** lightweight LLM call classifies query intent (direct/relational/multi-hop/interaction/contraindication) — used for logging/evaluation category tagging, not for changing correctness-critical logic.
2. **Entity identification:** extract candidate entity mentions from the question (same NER approach as Section 9, lighter weight since questions are short).
3. **Graph node matching:** fuzzy-match extracted mentions to existing graph nodes by `normalized_name` (Levenshtein/embedding similarity fallback for near-misses).
4. **Graph traversal:** from matched node(s), traverse outward using **parameterized, pre-approved Cypher templates** (Section 14) up to `MAX_HOP_DEPTH` (default 3).
5. **Multi-hop retrieval:** collect all nodes/edges along valid paths matching the query's implied relation types.
6. **Evidence collection:** for every edge traversed, pull its `source_id` provenance and retrieve the corresponding source chunk/document metadata.
7. **Optional vector retrieval:** if graph traversal returns too little (below a minimum evidence threshold), fall back to supplementing with vector search on the same corpus (this is the seam where Hybrid mode connects).
8. **Evidence fusion:** deduplicate overlapping evidence (same source cited by multiple edges), rank by confidence × relevance.
9. **LLM generation:** same shared prompt template as vector RAG (Section 15), evidence formatted as **graph paths + supporting text**, not flat chunks.
10. **Citations:** identical mechanism to vector RAG — resolve `source_id`s to full citation metadata.

**Traversal rules:**
- `MAX_HOP_DEPTH = 3` (configurable) — beyond 3 hops, evidence relevance drops sharply and noise increases; this is a deliberate, justified engineering bound, not a limitation to hide.
- Relationship selection restricted to types relevant to the classified query intent (e.g., an "interaction" question only traverses `INTERACTS_WITH` and adjacent `TREATS` edges, not unrelated `HAS_SYMPTOM` edges) — prevents combinatorial explosion.
- Graph filtering: exclude edges below the confidence threshold used at ingestion (Section 10) unless explicitly requested to show low-confidence evidence.
- Evidence ranking: primary by path length (shorter = more direct), secondary by edge confidence.
- Duplicate evidence: same source cited by multiple paths → collapse into one citation entry, list all paths it supports.

---

## SECTION 13 — HYBRID RETRIEVAL

**When vector retrieval is useful:** direct factual and general/semantic questions where a single well-matched chunk already contains the answer, and when graph coverage is sparse for the queried entity.

**When graph retrieval is useful:** relational, multi-hop, interaction, and contraindication questions — anything requiring explicit connection between entities.

**Merging strategy:** run both retrievers in parallel; if graph traversal returns ≥1 valid path with confidence above threshold, prioritize graph evidence and use vector results only as supplementary context (e.g., background chunk that mentions an entity but isn't part of a path). If graph traversal returns nothing, fall back to vector-only.

**Ranking:** graph-path evidence ranked above pure vector-similarity evidence when both exist for the same claim (graph evidence is explicit and structured, so it's treated as higher-confidence by design — this should also be tested empirically, not just assumed, in the ablation study).

**Deduplication:** if the same source document is retrieved by both paths, merge into a single citation.

**Optional reranking:** a cross-encoder reranker (e.g., `ms-marco-MiniLM` cross-encoder) can rerank the fused evidence list before it's passed to the LLM — recommended only if time remains after core pipelines work (see Section 27 ablations).

**Practical implementation:** implement as a single `hybrid_retrieve(query)` function that internally calls `graph_retrieve()` and `vector_retrieve()`, then applies the fusion rule above — keep vector-only and graph-only modes independently callable too, since the evaluation framework needs to run all three conditions separately.

---

## SECTION 14 — CYPHER QUERY STRATEGY

**Safety principle: no unrestricted LLM-generated Cypher.** The LLM never writes raw Cypher against the live database. Instead, the LLM (or a rule-based classifier) selects **which pre-written, parameterized template** to use and supplies only entity-name parameters, which are bound via Neo4j parameters (never string-concatenated) — this eliminates Cypher injection risk entirely.

**Example templates:**

One-hop:
```cypher
MATCH (d:Drug {normalized_name: $entity})-[r:TREATS]->(c:Condition)
RETURN d, r, c
```

Two-hop:
```cypher
MATCH (c:Condition {normalized_name: $condition})<-[:TREATS]-(d:Drug)-[r:CAUSES]->(se:SideEffect)
RETURN d, r, se
```

Multi-hop (bounded):
```cypher
MATCH path = (a {normalized_name: $entity_a})-[*1..3]-(b {normalized_name: $entity_b})
RETURN path LIMIT 10
```

Drug interaction:
```cypher
MATCH (a:Drug {normalized_name: $drug_a})-[r:INTERACTS_WITH]-(b:Drug {normalized_name: $drug_b})
RETURN a, r, b
```

Contraindications:
```cypher
MATCH (d:Drug {normalized_name: $drug})-[r:CONTRAINDICATED_FOR]->(target)
RETURN d, r, target
```

Evidence/source retrieval:
```cypher
MATCH (a)-[r]->(b)
WHERE id(r) = $rel_id
RETURN r.source_id AS source_id
```

Path retrieval / subgraph extraction (for the "why this answer" trace):
```cypher
MATCH path = shortestPath((a {normalized_name: $entity_a})-[*..3]-(b {normalized_name: $entity_b}))
RETURN path
```

**NL question → graph query conversion:** LLM classifies intent + extracts entity names (structured JSON output) → backend maps intent to the matching template above → template executed with the extracted names as bound parameters. If intent classification is ambiguous, backend tries the multi-hop generic template as a safe fallback and shows lower-confidence results clearly labeled as such.

---

## SECTION 15 — LLM GENERATION

**System prompt template (shared by both pipelines):**

```
You are a clinical literature research assistant. You are NOT a doctor and this is NOT medical advice.

You will be given a QUESTION and a set of EVIDENCE items, each with a source citation.

Rules:
1. Answer using ONLY the provided evidence. Do not use outside knowledge.
2. If the evidence is insufficient to answer, say so explicitly: "I do not have sufficient evidence in the corpus to answer this."
3. Clearly distinguish stated evidence from any inference you make connecting multiple evidence items ("Evidence states X. Combining with Y, this suggests... though this specific combination is not directly stated in a single source.").
4. Cite every claim with its source_id in [brackets].
5. Never provide a diagnosis, treatment recommendation, or dosage for an individual.
6. If evidence sources conflict, present both and say so — do not silently resolve the conflict.
7. Express uncertainty proportionate to evidence confidence and path length.

QUESTION: {question}
EVIDENCE: {evidence_items}
```

This prompt is identical for vector RAG and GraphRAG runs — only `{evidence_items}` differs in structure (flat chunks vs. graph paths) — required for a fair comparison.

---

## SECTION 16 — SOURCE PROVENANCE AND CITATIONS

**Metadata tracked per citation:** `document_id`, `title`, `source_type`, `publisher/journal`, `authors[]`, `year`, `section`, `url` (only if legally reusable), `text_span`, `extraction_method`.

**UI display:** inline numbered citation markers `[1]`, `[2]` in the answer text; a collapsible "Sources" panel below the answer listing full metadata per marker; clicking a marker scrolls to/highlights the corresponding source card.

---

## SECTION 17 — EXPLAINABILITY / REASONING PATH ("Why this answer?")

Displayed trace: `Question → identified entities → graph relationships used → traversal path (visual) → supporting documents → evidence snippets → final answer`.

**What this IS:** a factual trace of retrieval — which entities were matched, which edges were traversed, which documents were pulled, in what order.

**What this is NOT:** the LLM's hidden chain-of-thought. Never expose or fabricate an internal "reasoning narrative" for the LLM itself — only show the deterministic retrieval trace (graph paths, retrieved chunks), described using the terms **"evidence trace," "retrieval path," or "provenance"** — never "reasoning" or "thought process," per the project's safety rules (Section 23, Section 42).
