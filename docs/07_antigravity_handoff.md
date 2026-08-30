# 07 — Antigravity Implementation Handoff

## SECTION 40 — ANTIGRAVITY IMPLEMENTATION SPECIFICATION

1. **Project objective:** Implement the GraphRAG-Based Clinical Knowledge Retrieval and Decision Support System per this spec pack (files 01–06 in this `docs/` folder).
2. **Non-negotiable requirements:**
   - No real patient data, anywhere, ever
   - Every graph edge must carry provenance (`source_id`) — no edge without a traceable source
   - No unrestricted LLM-generated Cypher — parameterized templates only (Section 14)
   - Every generated answer must be evidence-grounded with citations, or explicitly state insufficient evidence
   - Safety disclaimer present on every answer + persistently in UI
   - Vector RAG baseline and GraphRAG must share corpus, chunking, LLM, and prompt template for a fair comparison
3. **Architecture:** Section 22 (ingestion → graph/vector store → retrieval layer → generation → API → frontend)
4. **Tech stack:** Python/FastAPI backend, Neo4j graph, FAISS/Qdrant vector store, sentence-transformers embeddings, React/TypeScript/Vite/Tailwind frontend, Cytoscape.js or React Flow for graph viz
5. **Directory structure:** Section 21 — follow exactly, do not restructure without documenting why in `docs/architectural_decisions.md`
6. **Data schemas:** Section 20 — all modules must use these exact field names/types
7. **Graph schema:** Section 6 — node/relationship types and constraints are closed-set; do not add new node/relationship types without updating Section 6 and getting the closed-set Cypher templates (Section 14) updated to match
8. **API schema:** Section 19
9. **Retrieval workflow:** Sections 11–13
10. **Evaluation workflow:** Sections 25–27
11. **Frontend requirements:** Section 18
12. **Testing requirements:** Section 31
13. **Security requirements:** Section 23
14. **Coding conventions:** type hints throughout Python (Pydantic models for all API I/O), TypeScript strict mode on frontend, no `any` types without justification comment, docstrings on all pipeline stage functions
15. **Environment variables:** Section 43
16. **Definition of done:** Section 44
17. **Phase-by-phase implementation order:** follow Section 29's 5 phases in order — do not begin frontend work before Phase 3 (retrieval) is functional, do not begin evaluation before Phase 2 (graph population) is complete

**Explicit prohibitions for the coding agent:**
- Do not replace core architecture (e.g., swapping GraphRAG for a different pattern) without explicit justification documented in `docs/architectural_decisions.md`
- Do not fabricate datasets or medical facts — if a source document is missing/unavailable, flag it as a blocker, do not synthesize placeholder "facts"
- Do not silently omit research components (baseline comparison, evaluation, ablations) — if a component is deferred due to time, log it explicitly as a known gap in `docs/known_gaps.md`
- Do not fabricate evaluation numbers — if evaluation hasn't been run yet, leave results as `TBD`, never invent plausible-looking numbers
- Do not expose hidden chain-of-thought in the UI — only evidence traces/retrieval paths (Section 17)
- Do not claim medical diagnosis capability anywhere in UI copy, docs, or prompts
- Do not implement unrestricted Cypher generation from LLM output (Section 14, Section 23)
- Write tests for all core pipeline functions (Section 31) — do not skip testing to save time
- Keep all configurable values (hop depth, top-K, confidence thresholds, model names) in config/environment files, never hardcoded
- Document architectural decisions as they're made, not retroactively

---

## SECTION 41 — PHASE-BY-PHASE ANTIGRAVITY PROMPTS

Each prompt below is meant to be sent to Antigravity **one at a time**, in order. Each includes objective, allowed/forbidden files, expected outputs, acceptance criteria, and a reminder to run tests and report blockers.

**Prompt 1 — Initialize repository and architecture**
> Objective: scaffold the repository per `docs/04_frontend_backend_architecture.md` Section 21's directory structure. Create empty module files with docstrings describing their responsibility, `docker-compose.yml` with `neo4j`, `backend`, `frontend` services, `.env.example` per Section 43, and a root `README.md` skeleton per Section 33.
> May change: entire repo (new project).
> Must not change: nothing yet exists — no constraint.
> Expected outputs: full folder tree, empty-but-documented modules, working `docker-compose up` (services start, even if backend/frontend are placeholder).
> Acceptance criteria: `docker-compose up` runs without crash; folder structure matches Section 21 exactly.
> Run: `docker-compose config` to validate compose file. Report any missing dependencies as blockers.

**Prompt 2 — Implement document ingestion (parsing)**
> Objective: implement `ingestion/parsers/` for PDF, JSON (openFDA), and XML (PubMed) source types per Section 8.
> May change: `ingestion/parsers/*`, `tests/ingestion/*`.
> Must not change: graph schema, API, frontend.
> Expected outputs: each parser takes a raw file path, returns cleaned plain text + metadata dict (Section 8 metadata schema).
> Acceptance criteria: parses ≥5 fixture documents (one per source type minimum) without crash; unit tests pass.
> Run tests: `pytest tests/ingestion/`. Report any source-type parsing edge cases as blockers.

**Prompt 3 — Implement chunking and metadata**
> Objective: implement `ingestion/chunking/` per Section 8's chunking strategy (section-aware fallback to sliding window, 700 tokens/100 overlap).
> May change: `ingestion/chunking/*`, `backend/app/models/` (Chunk schema).
> Must not change: parsers from Prompt 2.
> Expected outputs: chunker takes cleaned text + metadata, returns list of `Chunk` objects (Section 20 schema).
> Acceptance criteria: chunk offsets are correct and non-overlapping-except-declared-overlap; unit tests pass.
> Run tests: `pytest tests/ingestion/test_chunking.py`. Report blockers.

**Prompt 4 — Implement entity extraction**
> Objective: implement `ingestion/extraction/entity_extractor.py` per Section 9 (hybrid scispaCy/spaCy + LLM classification).
> May change: `ingestion/extraction/entity_extractor.py`, related tests.
> Must not change: chunking, parsers.
> Expected outputs: given a chunk, returns list of `Entity` objects matching Section 20 schema, typed only as one of the 7 schema node types.
> Acceptance criteria: runs on fixture chunks, produces schema-conformant output; unit tests validate schema conformance and reject invalid types.
> Run tests + report LLM API cost/latency observed as a note (not a blocker unless prohibitive).

**Prompt 5 — Implement relation extraction**
> Objective: implement `ingestion/extraction/relation_extractor.py` per Section 10 — closed relation-type set only.
> May change: `ingestion/extraction/relation_extractor.py`, related tests.
> Must not change: entity extractor interface (consumes its output).
> Expected outputs: given a chunk + its extracted entities, returns list of `Relationship` objects (Section 20 schema), rejecting any relation type outside the allowed list.
> Acceptance criteria: schema-conformance tests pass; a test asserting an out-of-schema relation type is rejected, not silently coerced.
> Run tests, report blockers.

**Prompt 6 — Implement extraction validation**
> Objective: implement `ingestion/validation/` per Sections 9–10 (dedup, confidence threshold, provenance requirement, contradiction flagging).
> May change: `ingestion/validation/*`.
> Must not change: extractors' output schema.
> Expected outputs: validator function takes raw extracted entities/relations, returns validated set + `validation_errors.log` entries for rejects.
> Acceptance criteria: rejects relations missing `source_id`; deduplicates by normalized name; flags (does not silently resolve) contradictory relations.
> Run tests, report blockers.

**Prompt 7 — Implement Neo4j graph creation**
> Objective: implement `backend/app/graph/` insertion logic per Section 8's graph insertion stage, using `MERGE` for idempotency.
> May change: `backend/app/graph/insert.py`, related tests.
> Must not change: validation module interface.
> Expected outputs: validated entities/relations inserted into Neo4j; re-running insertion on the same data does not create duplicates.
> Acceptance criteria: idempotency test passes (insert twice, node/edge counts unchanged after 2nd run).
> Run tests against a test Neo4j instance (Docker), report blockers.

**Prompt 8 — Implement graph schema and indexes**
> Objective: implement constraint/index creation per Section 6's Cypher constraints, run automatically on backend startup.
> May change: `backend/app/graph/schema.py`.
> Must not change: node/relationship types (fixed per Section 6).
> Expected outputs: constraints created idempotently (`IF NOT EXISTS`) on startup.
> Acceptance criteria: constraint violation (duplicate ID) is rejected by Neo4j, verified by test.
> Run tests, report blockers.

**Prompt 9 — Implement vector baseline RAG**
> Objective: implement `backend/app/retrieval/vector_retrieve.py` per Section 11.
> May change: retrieval module, vector store setup (FAISS/Qdrant), embedding indexing script.
> Must not change: chunking output schema.
> Expected outputs: given a query, returns top-K chunks with similarity scores.
> Acceptance criteria: retrieval test against fixture corpus returns expected chunk for a known query; latency logged.
> Run tests, report blockers.

**Prompt 10 — Implement GraphRAG retrieval**
> Objective: implement `backend/app/retrieval/graph_retrieve.py` per Section 12 — entity linking, template-based Cypher traversal, evidence collection.
> May change: retrieval module, Cypher templates (Section 14).
> Must not change: graph schema, vector retrieval.
> Expected outputs: given a query, returns traversed path(s) + evidence with source_ids.
> Acceptance criteria: fixture test (Section 31 example) — known 2-hop question returns correct path and citation.
> Run tests, report blockers.

**Prompt 11 — Implement hybrid retrieval**
> Objective: implement `backend/app/retrieval/hybrid_retrieve.py` per Section 13's fusion rule.
> May change: hybrid module only.
> Must not change: vector_retrieve, graph_retrieve interfaces.
> Expected outputs: fused, deduplicated, ranked evidence combining both.
> Acceptance criteria: test confirms graph evidence is prioritized when present, vector fallback triggers when graph returns nothing.
> Run tests, report blockers.

**Prompt 12 — Implement LLM answer generation**
> Objective: implement `backend/app/generation/` using the shared prompt template (Section 15) for all three retrieval modes.
> May change: generation module.
> Must not change: retrieval module interfaces.
> Expected outputs: given evidence + question, returns grounded answer with inline citation markers.
> Acceptance criteria: test with fixture evidence confirms output cites provided source_ids and doesn't hallucinate uncited claims (string-check for citation markers present).
> Run tests, report blockers.

**Prompt 13 — Implement citations/evidence provenance**
> Objective: implement `/api/evidence/{source_id}` resolution and inline citation-to-metadata mapping (Section 16).
> May change: API route + evidence resolver module.
> Must not change: generation module.
> Expected outputs: citation markers resolve to full metadata objects.
> Acceptance criteria: API test confirms correct metadata returned for known source_id, 404 for unknown.
> Run tests, report blockers.

**Prompt 14 — Implement evaluation framework**
> Objective: implement `evaluation/` per Section 25 — metrics calculators, per-category aggregation.
> May change: `evaluation/*`.
> Must not change: retrieval/generation modules (evaluation calls them, doesn't modify them).
> Expected outputs: given a benchmark set + mode, produces `EvaluationResult` objects (Section 20 schema).
> Acceptance criteria: metric calculators tested against hand-computed fixture examples (known precision/recall values).
> Run tests, report blockers.

**Prompt 15 — Implement benchmark dataset framework**
> Objective: create `evaluation/benchmark/benchmark_set.json` structure per Section 26, with placeholder entries to be filled once corpus is real.
> May change: benchmark JSON file, loader script.
> Must not change: evaluation metric logic.
> Expected outputs: loadable benchmark set with all 9 categories represented.
> Acceptance criteria: loader validates schema; flags any entry missing `expected_evidence` or `expected_answer`.
> Report which benchmark entries still need real evidence filled in (this is expected to be a manual/semi-manual task, not fully automatable).

**Prompt 16 — Implement FastAPI backend (full integration)**
> Objective: wire all modules together behind the API defined in Section 19.
> May change: `backend/app/api/*`, `backend/main.py`.
> Must not change: individual module internals (only orchestration).
> Expected outputs: all endpoints from Section 19 functional.
> Acceptance criteria: `/api/health` returns true for all dependencies; `/api/query` full round trip works for all 3 modes.
> Run integration tests, report blockers.

**Prompt 17 — Implement React frontend (core pages)**
> Objective: build Query Interface, Answer Panel, Evidence Panel per Section 18.
> May change: `frontend/src/*`.
> Must not change: backend API contracts.
> Expected outputs: functional query→answer→evidence flow against the live backend.
> Acceptance criteria: manual test — submitting a question renders answer + citations.
> Report any API contract mismatches as blockers (do not silently adapt the API instead of flagging it).

**Prompt 18 — Implement graph visualization**
> Objective: build Graph Visualization + Retrieval Trace components per Section 18, using Cytoscape.js or React Flow, consuming `/api/graph/neighborhood`.
> May change: `frontend/src/components/GraphView*`.
> Must not change: other frontend pages, backend.
> Expected outputs: traversal path renders visually after an answer is returned.
> Acceptance criteria: manual test — graph mode answer shows the correct path highlighted.
> Report rendering performance issues if graph is large.

**Prompt 19 — Implement end-to-end integration**
> Objective: connect all frontend pages, add loading/error states per Section 24's error handling table.
> May change: frontend + minor backend error-response adjustments.
> Must not change: core architecture.
> Expected outputs: full app usable start-to-finish including error states (Neo4j down, insufficient evidence, etc.).
> Acceptance criteria: manually trigger each error condition from Section 24, confirm correct UI behavior.
> Report blockers.

**Prompt 20 — Implement testing**
> Objective: fill out remaining test coverage per Section 31 across all modules, add e2e tests (Playwright/Cypress).
> May change: `tests/*`.
> Must not change: application code (fix only if a test reveals a genuine bug, and document the fix).
> Expected outputs: test suite covering unit/integration/API/frontend/e2e per Section 31.
> Acceptance criteria: `pytest` and frontend test suite both pass; e2e test completes a full query flow.
> Run full test suite, report failures/blockers.

**Prompt 21 — Implement Docker/deployment**
> Objective: finalize `docker-compose.yml` for full-stack local deployment per Section 32.
> May change: `docker-compose.yml`, Dockerfiles.
> Must not change: application logic.
> Expected outputs: fresh clone + `docker-compose up` brings up a fully working system.
> Acceptance criteria: verified on a clean environment (no leftover local state).
> Report blockers.

**Prompt 22 — Perform final audit against requirements**
> Objective: cross-check the full implementation against Section 44's Definition of Done, line by line.
> May change: nothing without flagging first — this is an audit pass.
> Expected outputs: a `docs/audit_report.md` listing each Definition-of-Done item as ✅/❌/⚠️ with notes.
> Acceptance criteria: every item is addressed or explicitly logged as a known gap in `docs/known_gaps.md`.
> Report final blockers and recommend next steps (e.g., "evaluation numbers still TBD pending real corpus ingestion").

---

## SECTION 42 — ANTIGRAVITY RULES FOR THE WHOLE PROJECT (save as `docs/project_rules.md` / agent instruction file)

**Coding principles:** type-safe (Pydantic + TypeScript strict), small composable functions per pipeline stage, docstrings on all pipeline functions, no dead code.

**Architecture principles:** follow Section 21/22 exactly; any deviation requires a documented reason in `docs/architectural_decisions.md`; don't add node types, relationship types, or services not justified by actual data/requirements.

**Safety principles:** no unrestricted Cypher from LLM output (parameterized templates only); no real patient data ever; persistent safety disclaimer; prefer "insufficient evidence" over guessing, always.

**Testing rules:** every pipeline stage gets unit tests before being considered done; integration tests for cross-module flows; no merging/completing a phase without its tests passing.

**Documentation rules:** update README/docs as features land, not retroactively at the end; log architectural decisions and known gaps as they occur.

**No fake data/results:** never fabricate benchmark evidence, citations, or evaluation numbers. If data isn't available yet, mark fields as `TBD` explicitly.

**No unnecessary dependencies:** justify every new library addition against an actual pipeline need (Section 8-15).

**No over-engineering:** no microservices, no unnecessary cloud infra, no schema entities without extracted-data justification (Section 1's explicit out-of-scope list is binding).

**Maintainability:** consistent naming across ingestion/graph/API/frontend layers (Section 20's schemas are the single source of truth for field names).

**Reproducibility:** `docker-compose up` + documented ingestion command must reproduce the full system from a fresh clone.

**Security:** Section 23 is binding — especially Cypher parameterization and secrets-in-env-only.

**Medical-domain caution:** every UI surface, doc, and prompt must reflect that this is a non-deployable research prototype, never framed as clinical-decision-ready.

---

## SECTION 43 — ENVIRONMENT CONFIGURATION (`.env.example`)

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme

# LLM
LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=your_model_name_here

# Embeddings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Vector store
VECTOR_STORE_TYPE=faiss   # or qdrant
VECTOR_STORE_PATH=./data/vector_index

# Retrieval config
MAX_HOP_DEPTH=3
VECTOR_TOP_K=5
RELATION_CONFIDENCE_THRESHOLD=0.6

# Application
BACKEND_PORT=8000
FRONTEND_PORT=5173
LOG_LEVEL=info
```

*No real secrets are placed in this file — placeholders only, exactly as required by Section 43's original instructions.*

---

## SECTION 44 — ACCEPTANCE CRITERIA (Definition of Done)

The project is **not complete** unless every item below is true:

- [ ] Documents can be ingested (parsing → cleaning → chunking works end-to-end)
- [ ] Entities are extracted and schema-validated
- [ ] Relations are extracted and schema-validated (closed relation-type set enforced)
- [ ] Graph is populated in Neo4j meeting minimum dataset scope (Section 5)
- [ ] Graph queries work via parameterized Cypher templates (Section 14)
- [ ] Baseline vector RAG works, returns grounded + cited answers
- [ ] GraphRAG works, returns traversal-based grounded + cited answers
- [ ] Hybrid retrieval works, fuses both correctly
- [ ] Every answer has evidence and citations, or explicitly states insufficient evidence
- [ ] Graph traversal paths are displayed in the frontend
- [ ] Benchmark questions run against all 3 modes
- [ ] Standard RAG vs GraphRAG comparison runs and produces real (not fabricated) metrics
- [ ] Metrics are computed from actual experiment runs, stored, and reportable
- [ ] Tests pass (unit, integration, API, frontend, e2e)
- [ ] Frontend works end-to-end (query → answer → evidence → graph)
- [ ] Backend works end-to-end, all endpoints functional
- [ ] Documentation complete (README, architectural decisions, known gaps)
- [ ] Safety disclaimer present on every answer and persistently in UI

---

## SECTION 45 — FINAL MASTER SPECIFICATION (handoff summary)

- **Project title:** GraphRAG-Based Clinical Knowledge Retrieval and Decision Support System
- **Objective:** Build and empirically compare vector RAG vs GraphRAG vs hybrid retrieval on a hand-built clinical knowledge graph from public literature.
- **Research question:** Does graph-based retrieval improve multi-hop clinical QA vs conventional vector RAG, and for which question categories?
- **Scope:** Public literature only, no patient data, no diagnostic capability, bounded graph schema (7 node types, 6 relation types), bounded hop depth (3).
- **Architecture:** Ingestion (parse→clean→chunk→extract→validate) → Neo4j graph + vector store → 3 retrieval modes (vector/graph/hybrid) → shared LLM generation prompt → FastAPI → React/Tailwind frontend with graph visualization and evidence trace.
- **Stack:** Python/FastAPI/Neo4j/sentence-transformers/FAISS-or-Qdrant backend; React/TypeScript/Vite/Tailwind/Cytoscape.js-or-React-Flow frontend; Docker Compose deployment.
- **Data strategy:** openFDA labels (anchor), open-access PubMed/PMC articles, 1–2 open clinical guidelines; 30–80 documents; no patient data.
- **Graph schema:** Drug, Condition, Symptom, SideEffect, Population, ClinicalStudy, Guideline nodes; TREATS, CAUSES, HAS_SYMPTOM, INTERACTS_WITH, CONTRAINDICATED_FOR, RECOMMENDS relations; every edge requires a `source_id`.
- **Retrieval methods:** vector (embedding similarity, top-K), graph (entity linking + parameterized Cypher traversal, max 3 hops), hybrid (graph-prioritized fusion with vector fallback).
- **Generation method:** shared strict-grounding prompt (Section 15) across all modes, citation-required, insufficient-evidence-preferred-over-hallucination.
- **Evaluation:** ~60–80 question benchmark across 9 categories; precision/recall/faithfulness/hallucination-rate/multi-hop-accuracy metrics; manual scoring as primary source of truth, LLM-as-judge as secondary.
- **Frontend:** query interface, answer + evidence + graph-visualization + retrieval-trace panels, evaluation dashboard.
- **Backend:** FastAPI, 8 endpoints (Section 19), Pydantic-validated I/O.
- **Testing:** unit/integration/API/frontend/e2e per Section 31, mandatory before each phase is considered complete.
- **Deployment:** local Docker Compose (Neo4j + backend + frontend); no cloud infra required.
- **Safety restrictions:** no patient data, no diagnostic claims, no unrestricted Cypher generation, persistent disclaimer, evidence-trace (not chain-of-thought) explainability only.
- **Timeline:** 5 phases (Section 29), assumed ~10 weeks total, rescale proportionally to actual available time.
- **Definition of done:** Section 44, all items checked before considering the project complete.
