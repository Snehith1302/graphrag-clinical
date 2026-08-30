# 05 — Safety, Evaluation, Timeline, Testing, Deployment

## SECTION 23 — SECURITY AND SAFETY

- **API key handling:** all keys (LLM API, Neo4j credentials) loaded from environment variables only, never hardcoded, never logged
- **Secrets management:** `.env` file (gitignored), `.env.example` committed with placeholders only (Section 43)
- **Input validation:** Pydantic schemas validate all API request bodies; question length capped (e.g., 1000 chars) to bound cost/latency
- **Prompt injection considerations:** treat retrieved document text as untrusted content — system prompt explicitly instructs the LLM to follow only the system instructions, not any instruction-like text found inside retrieved evidence; strip/flag documents containing suspicious instruction-like patterns during ingestion
- **Malicious document considerations:** ingestion pipeline runs in a sandboxed step with file-type/size limits; PDFs parsed with a library that doesn't execute embedded scripts
- **Cypher injection prevention:** absolute rule — all Cypher execution uses parameterized queries via the Neo4j driver's parameter binding; **no string concatenation into Cypher, ever**, and no LLM-generated raw Cypher is executed directly (Section 14)
- **Rate limiting:** basic per-IP rate limit on `/api/query` (e.g., 20 req/min) to bound LLM API cost during demos
- **Logging:** log requests/errors, never log full LLM API keys or `.env` contents; redact if needed
- **Privacy:** no real patient data exists in the system by design — nothing to protect at that level, but treat all ingested text as attributed-source content requiring citation, not personal data
- **Medical safety disclaimer (must appear on every answer and persistently in UI):**
  > "This is a research prototype for literature retrieval only. It is not a medical device, does not provide diagnosis or treatment advice, and must not be used for real clinical decisions. Consult a qualified healthcare professional for medical guidance."

## SECTION 24 — ERROR HANDLING

| Condition | Behavior |
|---|---|
| No documents ingested yet | `/api/query` returns `insufficient_evidence` with a clear message, not an error |
| Malformed document during ingestion | Log, skip, continue with remaining documents; report skipped count |
| Failed entity extraction (per chunk) | Log, skip chunk's entities, continue pipeline |
| Failed relation extraction (per chunk) | Same — isolated failure, doesn't block document or run |
| Graph unavailable (Neo4j down) | Query falls back to vector-only mode automatically, with a UI note: "Graph retrieval unavailable — showing vector-only results" |
| Vector search unavailable | Falls back to graph-only mode with equivalent UI note |
| LLM unavailable | Return 503 with retry guidance; do not fabricate an answer without LLM |
| Insufficient evidence | Explicit answer: "I do not have sufficient evidence in the corpus to answer this." (never silently guess) |
| Conflicting evidence | Present both sources, flag conflict explicitly (Section 10) |
| Irrelevant question (off-topic) | LLM responds that the system only covers clinical literature in its corpus |
| Unsupported medical question (personalized advice) | Explicit decline + redirect to professional (Use Case 8, Section 4) |

**Guiding rule, restated:** the system always prefers "I do not have sufficient evidence" over hallucination — this is tested directly in evaluation (hallucination rate metric).

## SECTION 25 — EVALUATION FRAMEWORK

**Retrieval metrics:**
- **Precision@K:** fraction of top-K retrieved evidence items that are relevant (human/rubric-labeled) to the question
- **Recall@K:** fraction of all relevant evidence items (from the benchmark's expected evidence set) that appear in top-K retrieved
- **MRR:** if a question has one clearly-best evidence item, rank of its first appearance

**Answer metrics:**
- **Correctness:** does the answer match the expected answer (human-rubric-scored, 0/0.5/1)
- **Faithfulness:** is every claim in the answer traceable to cited evidence (checked manually or via an LLM-as-judge cross-check against retrieved evidence, clearly labeled as an automated proxy, not ground truth)
- **Completeness:** does the answer cover all expected sub-facts for multi-hop questions
- **Citation correctness:** do the cited `source_id`s actually support the claim they're attached to
- **Hallucination rate:** fraction of answers containing a claim not traceable to any retrieved evidence

**Multi-hop metrics:**
- **Entity identification accuracy:** did the system correctly identify all entities named/implied in the question
- **Relationship path accuracy:** for graph mode, did the traversed path match the expected path in the benchmark
- **Multi-hop answer accuracy:** correctness specifically on 2-hop+ benchmark questions

**System metrics:** end-to-end latency, retrieval time, generation time, token consumption (if API-based LLM)

**Calculation approach:** each benchmark question has a hand-authored "expected evidence set" (list of source_ids) and "expected answer" (short reference answer). Precision/Recall are computed by set comparison against retrieved `source_ids`. Correctness/faithfulness/completeness use a rubric scored by the student (manual) supplemented by an LLM-as-judge pass — both scores reported, with manual scoring as the primary source of truth (LLM-as-judge alone would be circular/self-grading risk since the same LLM generates answers).

## SECTION 26 — BENCHMARK DATASET

**Categories:** direct, semantic, relationship, two-hop, multi-hop, interaction, contraindication, source attribution, unanswerable

**Recommended size:** ~60–80 questions total, roughly 7–9 per category — enough for category-level trend analysis within a bounded corpus, not claimed as statistically definitive.

**10 sample benchmark questions (structure shown; actual evidence must be filled in against your real ingested corpus — do not fabricate):**

| # | Category | Question | Expected evidence path (placeholder) |
|---|---|---|---|
| 1 | Direct | "What is [Drug A] indicated for?" | `[doc: label_XXX, section: Indications]` |
| 2 | Semantic | "What treatments exist for [Condition X]?" | `[doc: guideline_XXX]` |
| 3 | Relationship | "What side effects does [Drug A] cause?" | `[doc: label_XXX, section: Adverse Reactions]` |
| 4 | Two-hop | "What side effects are linked to drugs treating [Condition X]?" | `[doc A: TREATS, doc B: CAUSES]` |
| 5 | Multi-hop | "Do drugs for [Condition X] and [Condition Y] have a known interaction?" | `[doc A, doc B, doc C — cross-referenced]` |
| 6 | Interaction | "Does [Drug A] interact with [Drug B]?" | `[doc: study_XXX]` |
| 7 | Contraindication | "Is [Drug A] contraindicated in [Population]?" | `[doc: label_XXX, section: Contraindications]` |
| 8 | Source attribution | "What is the source for [claim]?" | `[doc: exact citation]` |
| 9 | Unanswerable (out of corpus) | "What's the interaction between [Drug not in corpus] and [Drug A]?" | `expected: "insufficient evidence"` |
| 10 | Unanswerable (personalized) | "What dose should I give a 70kg patient with [Condition]?" | `expected: decline + redirect` |

All `doc_XXX` placeholders must be replaced with real document IDs once your actual corpus is ingested — never fabricate citation contents in the benchmark file itself.

## SECTION 27 — ABLATION STUDIES

| Experiment | What it demonstrates |
|---|---|
| Vector-only vs Graph-only vs Hybrid | Core research question — which retrieval strategy wins per category |
| With/without reranking (hybrid) | Whether a cross-encoder rerank step meaningfully improves fused evidence quality |
| Hop depth = 1 vs 2 vs 3 | Whether deeper traversal improves recall or just adds noise (precision/recall tradeoff curve) |
| Top-K = 3 vs 5 vs 10 (vector) | Standard RAG tuning sensitivity, contextualizes the baseline's fairness |
| Confidence threshold = 0.5 vs 0.7 vs 0.9 (graph edges) | Precision/recall tradeoff on graph evidence inclusion |

## SECTION 28 — EXPECTED RESULTS

**Do not fabricate numbers.** State only trends and success/failure criteria:

- **Expected trend:** GraphRAG should show measurably higher Recall@K and multi-hop answer accuracy on two-hop/multi-hop/interaction/contraindication categories; roughly comparable performance to vector RAG on direct/semantic categories.
- **What counts as success:** a statistically/practically meaningful gap (e.g., ≥15–20 percentage points, though the exact threshold should be decided before running experiments, not after) favoring GraphRAG specifically on relational categories, with hybrid matching-or-exceeding the better of the two on every category.
- **What counts as a negative result:** no meaningful difference between methods, or GraphRAG underperforming due to graph sparsity/extraction noise — this is still a valid, reportable finding (e.g., "graph benefits are contingent on extraction quality/coverage, which was a bottleneck in this bounded corpus").
- **Reporting inconclusive findings:** report exact numbers with confidence caveats ("on this 70-question benchmark over a X-document corpus, results suggest... larger-scale validation would be needed to generalize").

## SECTION 29 — IMPLEMENTATION PLAN (Phase-based, scale to your real timeline)

> Built in 5 phases. At the assumed 10-week pace this is ~2 weeks/phase; if you have more or less time, stretch/compress each phase proportionally rather than skipping steps.

**Phase 1 — Foundation (Data + Graph skeleton)**
- Objectives: repo scaffolded, source documents collected, ingestion pipeline (parsing→chunking) working, Neo4j schema/constraints created
- Deliverables: `ingestion/` pipeline runs end-to-end on ≥5 test documents; empty graph schema live in Neo4j
- Testing: unit tests for parsers/chunkers

**Phase 2 — Extraction + Graph Population**
- Objectives: entity + relation extraction working with validation; graph populated from full document set
- Deliverables: populated Neo4j graph meeting minimum dataset scope (Section 5); `validation_errors.log` reviewed and extraction quality spot-checked manually
- Testing: extraction schema-conformance tests, spot-check accuracy sample (manual review of ~30 extracted relations)

**Phase 3 — Retrieval Pipelines**
- Objectives: vector RAG baseline working; GraphRAG traversal + Cypher templates working; hybrid fusion working
- Deliverables: all three modes return grounded answers with citations for hand-tested questions
- Testing: retrieval unit tests, Cypher template tests, manual QA on ~10 questions per mode

**Phase 4 — Evaluation + Frontend**
- Objectives: benchmark set finalized with real evidence paths; evaluation framework runs all 3 modes; frontend built (query, answer, evidence, graph viz, trace)
- Deliverables: evaluation results table/chart; working end-to-end UI
- Testing: API integration tests, frontend component tests, e2e test (ask question → see answer + graph)

**Phase 5 — Polish, Docs, Report**
- Objectives: ablation studies run; README/report/paper drafted; final bug fixes; Docker Compose verified
- Deliverables: final comparison writeup, complete documentation set, deployable Docker Compose stack
- Testing: full regression pass, fresh-clone `docker-compose up` verification

## SECTION 30 — MILESTONES AND CHECKPOINTS

1. Dataset pipeline works (parses + chunks real documents)
2. Knowledge graph populated (meets minimum scope, constraints enforced)
3. Baseline vector RAG works (grounded answers with citations)
4. GraphRAG works (traversal + evidence + citations)
5. Hybrid retrieval works
6. Evaluation framework works (runs benchmark, computes metrics)
7. Frontend works (full query→answer→evidence→graph flow)
8. Final Standard RAG vs GraphRAG comparison completed with real numbers

## SECTION 31 — TESTING STRATEGY

- **Unit tests:** chunker offsets, entity/relation JSON schema validators, Cypher template parameter binding
- **Integration tests:** ingestion pipeline end-to-end on a fixture document, `/api/query` full round trip against a test Neo4j instance
- **Graph DB tests:** constraint enforcement (duplicate ID rejected), MERGE idempotency (re-running ingestion doesn't duplicate nodes)
- **Retrieval tests:** given a fixture graph, does `graph_retrieve()` return the expected path for a known 2-hop question
- **API tests:** each endpoint's request/response schema, error codes for malformed input
- **Frontend tests:** component rendering (React Testing Library) for AnswerPanel, EvidencePanel with mock data
- **End-to-end tests:** Playwright/Cypress — submit a question in the UI, assert answer + citation + graph panel render

**Example test case:** "Given a fixture graph with `Metformin -[CONTRAINDICATED_FOR]-> Severe Renal Impairment` (source: `label_test_01`), querying 'Is Metformin contraindicated in severe renal impairment?' in graph mode should return an answer containing `label_test_01` as a citation and no hallucinated additional contraindications."

## SECTION 32 — DEPLOYMENT STRATEGY

- **Local development:** `docker-compose up` running Neo4j, backend (FastAPI, `uvicorn --reload`), frontend (`vite dev`)
- **Docker Compose:** three services — `neo4j`, `backend`, `frontend`; backend depends on `neo4j` healthcheck; `.env` mounted for config
- **Neo4j:** official Neo4j Docker image, persistent volume for graph data, exposed on default ports (7474/7687) for local Browser access during development/debugging
- **Backend deployment:** containerized FastAPI, can run locally via Docker Compose for demo/viva; no need for cloud hosting unless the panel specifically expects a live public demo — a local `docker-compose up` demo is standard and sufficient for a student project
- **Frontend deployment:** Vite build served either via the same Docker Compose (nginx static serve) or run in dev mode for the demo — keep it simple, avoid unnecessary cloud infra per the project's own over-engineering constraint
