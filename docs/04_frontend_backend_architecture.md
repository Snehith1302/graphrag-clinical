# 04 — Frontend, Backend, Data Models, Directory Structure, Architecture

## SECTION 18 — FRONTEND SPECIFICATION (React + TypeScript + Tailwind)

**Pages/components:**
- **Landing/Dashboard:** project title, safety disclaimer banner (persistent, non-dismissible), graph stats summary (node/edge counts), "Ask a question" CTA
- **Query Interface:** text input, retrieval-mode selector (Vector / Graph / Hybrid — for demo/comparison purposes), submit button, loading state
- **Answer Panel:** generated answer with inline citation markers, confidence/evidence-sufficiency indicator, explicit "insufficient evidence" state styling (amber, not red — not an error, a valid outcome)
- **Evidence Panel:** list of source cards (title, type, year, excerpt), expandable
- **Graph Visualization:** Cytoscape.js or React Flow rendering the traversal path for the current answer — nodes colored by type, edges labeled by relation type
- **Source Details:** modal/drawer with full citation metadata on node/edge click
- **Retrieval Trace:** step list ("Matched entity: Metformin → Traversed CONTRAINDICATED_FOR → Found: Severe Renal Impairment")
- **Evaluation Dashboard:** (if benchmark has been run) table/chart comparing Vector vs Graph vs Hybrid metrics per category
- **Settings/About:** model config display (read-only), safety disclaimer, project description, links to report/paper

**Layout:** two-column on desktop — left: query + answer + evidence; right: graph visualization + retrieval trace. Collapses to stacked tabs on mobile.

**Interactions:** submitting a query triggers the backend `/query` call; answer streams in (or loads with a spinner); graph panel animates traversal path drawing after the answer resolves.

**Graph visualization behavior:** default view shows only nodes/edges relevant to the current answer's traversal path (not the whole graph — avoids visual clutter and keeps the demo readable). A "show full local neighborhood" toggle expands one extra hop around matched entities.

**Node/edge click behavior:** clicking a node opens Source Details showing all documents that mention this entity; clicking an edge opens the specific citation(s) backing that relationship.

---

## SECTION 19 — BACKEND API SPECIFICATION (FastAPI)

| Method | Path | Purpose | Request | Response | Errors |
|---|---|---|---|---|---|
| POST | `/api/query` | Submit a question, get an answer | `{question, mode: "vector"\|"graph"\|"hybrid"}` | `{answer, citations[], evidence_trace, confidence}` | 400 invalid mode, 503 LLM unavailable |
| GET | `/api/graph/neighborhood?entity=` | Get local graph neighborhood for visualization | query param `entity`, `hops` (default 1) | `{nodes[], edges[]}` | 404 entity not found |
| GET | `/api/evidence/{source_id}` | Get full citation metadata | path param | `{document metadata}` | 404 not found |
| POST | `/api/ingest` | Trigger document ingestion (admin/dev use) | `{file_path or upload}` | `{status, document_id, entities_extracted, relations_extracted}` | 400 unsupported format, 500 parsing failure |
| GET | `/api/graph/stats` | Graph statistics for dashboard | — | `{node_counts_by_type, edge_counts_by_type, total_documents}` | — |
| GET | `/api/health` | Health check | — | `{status: "ok", neo4j: bool, vector_store: bool, llm: bool}` | — |
| POST | `/api/evaluate/run` | Run benchmark evaluation (dev/eval use) | `{benchmark_set_id, modes: []}` | `{run_id, status}` (async) | 400 invalid benchmark set |
| GET | `/api/evaluate/results/{run_id}` | Get evaluation results | path param | `{metrics_by_mode, metrics_by_category}` | 404 not found |

**Authentication assumption:** none required for the student-project demo (no real patient data, public deployment risk is low); if deployed publicly, add a simple API key check on `/api/ingest` and `/api/evaluate/*` only (admin-ish endpoints), leave `/api/query` open for demo purposes.

---

## SECTION 20 — DATA MODELS / JSON SCHEMAS

```json
// Document
{"document_id":"str","title":"str","source_type":"guideline|study|label","publisher":"str","authors":["str"],"year":2024,"url":"str|null","ingestion_date":"iso8601"}

// Chunk
{"chunk_id":"str","document_id":"str","text":"str","start_offset":0,"end_offset":700,"section_title":"str|null","embedding_id":"str"}

// Entity
{"entity_id":"str","normalized_name":"str","entity_type":"Drug|Condition|Symptom|SideEffect|Population|ClinicalStudy|Guideline","confidence":0.9,"document_id":"str","source_span":[0,10]}

// Relationship
{"relation_id":"str","source_entity_id":"str","relation_type":"TREATS|CAUSES|HAS_SYMPTOM|INTERACTS_WITH|CONTRAINDICATED_FOR|RECOMMENDS","target_entity_id":"str","confidence":0.85,"source_ids":["str"]}

// Graph Node (API response shape)
{"id":"str","label":"Drug","name":"Metformin","properties":{}}

// Graph Edge (API response shape)
{"id":"str","source":"str","target":"str","type":"TREATS","properties":{"confidence":0.9,"source_ids":["str"]}}

// Query
{"question":"str","mode":"vector|graph|hybrid"}

// Retrieval Result
{"evidence_items":[{"type":"chunk|graph_path","content":"str","source_ids":["str"],"confidence":0.9}]}

// Evidence (citation)
{"source_id":"str","document_id":"str","title":"str","section":"str|null","excerpt":"str","url":"str|null"}

// Generated Answer
{"answer_text":"str","citations":[{"marker":1,"source_id":"str"}],"confidence":"high|medium|low|insufficient_evidence","mode_used":"vector|graph|hybrid"}

// Evaluation Result
{"run_id":"str","mode":"vector|graph|hybrid","category":"direct|relational|two_hop|multi_hop|interaction|contraindication|citation|unanswerable","precision_at_k":0.8,"recall_at_k":0.75,"faithfulness_score":0.9,"hallucination_flag":false,"latency_ms":1200}
```

All IDs (`document_id`, `entity_id`, `source_id`) are consistent strings used across ingestion, graph, vector store, and API layers — no ID translation layer needed.

---

## SECTION 21 — PROJECT DIRECTORY STRUCTURE

```
graphrag-clinical/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI route definitions
│   │   ├── retrieval/      # vector_retrieve, graph_retrieve, hybrid_retrieve
│   │   ├── generation/     # LLM prompt templates, generation logic
│   │   ├── graph/          # Neo4j driver, Cypher templates, schema constraints
│   │   ├── models/         # Pydantic schemas (matches Section 20)
│   │   └── config.py       # env-driven configuration
│   └── main.py
├── ingestion/
│   ├── parsers/            # PDF/JSON/XML parsers per source type
│   ├── chunking/
│   ├── extraction/         # entity_extractor.py, relation_extractor.py
│   └── validation/         # schema validators, dedup logic
├── evaluation/
│   ├── benchmark/          # benchmark question set (JSON)
│   ├── metrics/            # precision/recall/faithfulness calculators
│   └── run_evaluation.py
├── frontend/
│   ├── src/
│   │   ├── components/     # QueryInput, AnswerPanel, EvidencePanel, GraphView, etc.
│   │   ├── pages/
│   │   ├── api/            # typed API client
│   │   └── App.tsx
│   └── tailwind.config.js
├── data/
│   ├── raw/                # untouched source documents
│   └── processed/          # cleaned/chunked intermediate outputs
├── tests/
│   ├── backend/
│   ├── ingestion/
│   └── e2e/
├── docs/                   # this spec pack + README + report + paper
├── docker-compose.yml
└── .env.example
```

Each folder's responsibility maps 1:1 to a pipeline stage from Section 8/11/12 — no folder exists without a corresponding pipeline responsibility (avoids over-engineering).

---

## SECTION 22 — SOFTWARE ARCHITECTURE

**High-level architecture (ASCII):**
```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  React UI   │─────▶│  FastAPI     │─────▶│  Retrieval      │
│ (Vite+TS+   │◀─────│  Backend     │◀─────│  Layer          │
│  Tailwind)  │      │              │      │ (vector/graph/  │
└─────────────┘      └──────┬───────┘      │  hybrid)        │
                             │              └────────┬────────┘
                             │                        │
                    ┌────────▼────────┐      ┌────────▼────────┐
                    │  LLM Generation │      │  Neo4j Graph DB │
                    │  (API/local LLM)│      │  Vector Store   │
                    └─────────────────┘      │  (FAISS/Qdrant) │
                                              └─────────────────┘
```

**Ingestion flow:**
```
Raw Docs → Parser → Cleaner → Chunker → [Entity Extractor + Relation Extractor] → Validator → {Neo4j Insert, Vector Index}
```

**Query flow:**
```
User Question → Backend /api/query → Retrieval Layer (mode-dependent) → Evidence Fusion → LLM Generation → Response (answer + citations + trace) → Frontend renders Answer/Evidence/Graph panels
```

**Evaluation flow:**
```
Benchmark Set → For each question × each mode → Run Query Flow → Compare answer/evidence to expected → Compute metrics → Aggregate by category → Store Evaluation Result
```

**Request flow (query, detailed):**
```
Frontend POST /api/query
  → FastAPI route validates request
  → Retrieval Layer dispatches by mode
  → (graph/hybrid) Entity linking → Cypher template execution → Evidence Fusion
  → (vector/hybrid) Embedding → Vector search
  → Generation module builds prompt → calls LLM
  → Response assembled (answer + citations + evidence_trace)
  → Returned to frontend
```
