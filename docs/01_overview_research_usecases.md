# 01 — Executive Overview, Research Gap, Research Question, Use Cases

> **Timeline assumption (stated explicitly, no fixed deadline was given):** This spec pack assumes a **10-week accelerated schedule**, on the reasoning that Antigravity absorbs most scaffolding/CRUD/boilerplate work, freeing student time for the non-automatable parts: dataset curation, schema validation, extraction quality review, evaluation design, and report writing. If your real timeline differs, Section 29's week-by-week plan is built in **phases**, not fixed dates — compress or stretch each phase proportionally.

---

## SECTION 1 — EXECUTIVE PROJECT OVERVIEW

**Project Title:** GraphRAG-Based Clinical Knowledge Retrieval and Decision Support System

**One-line description:** A research prototype that builds a clinical knowledge graph from public medical literature and compares graph-based retrieval against conventional vector RAG for multi-hop clinical question answering.

**100-word overview:**
Conventional Retrieval-Augmented Generation (RAG) retrieves text chunks using semantic similarity, which works well for direct factual questions but struggles when an answer requires connecting information scattered across multiple documents — e.g., a drug's interaction with a condition mentioned in a different guideline than its side effects. This project builds a Neo4j knowledge graph from public clinical/biomedical literature, extracts entities and relationships using NLP/LLM-assisted pipelines, and implements graph-based multi-hop retrieval (GraphRAG) alongside a standard vector RAG baseline. Both systems answer the same benchmark questions; retrieval quality, faithfulness, and hallucination rate are measured and compared to test whether graph structure improves multi-hop clinical QA.

**Detailed project overview:**
The system ingests a curated set of publicly available clinical guidelines and open-access biomedical literature (no patient data). An extraction pipeline identifies clinical entities (drugs, conditions, symptoms, interactions) and relationships between them, populating a Neo4j graph with full source provenance on every node and edge. Two retrieval paths are built on top of this data: (1) a standard vector RAG baseline using chunk embeddings and cosine similarity, and (2) a GraphRAG pipeline that identifies query entities, traverses the graph up to a bounded hop depth, and collects evidence along relationship paths. A hybrid mode fuses both. An LLM generates answers strictly grounded in retrieved evidence, with citations and an explicit "evidence trace" UI showing the graph path used. A benchmark question set spanning direct, relational, two-hop, multi-hop, and unanswerable categories is used to evaluate both pipelines on retrieval precision/recall, answer faithfulness, and hallucination rate.

**Problem statement:**
Standard RAG treats documents as independent chunks and retrieves by similarity alone. It has no mechanism to explicitly connect Entity A (in Document 1) to Entity B (in Document 7) via a relationship that is never stated in a single passage. Clinical knowledge is inherently relational (drug–condition–interaction–population chains), so this limitation is especially costly in this domain.

**Motivation:**
Multi-hop reasoning failures are a well-documented weakness of vector RAG in current literature. Testing whether an explicit knowledge graph mitigates this — in a bounded, safe, non-clinical-deployment context — is a legitimate, scoped undergraduate research question with clear engineering deliverables and a genuine research comparison.

**Proposed solution:**
Build the graph, build both retrieval pipelines on shared data and shared LLM, run identical benchmark questions through both, and measure the difference quantitatively rather than asserting it.

**Target users:** Not real clinicians or patients — the target "user" is a researcher/evaluator/reviewer exploring clinical literature. This must be explicit everywhere in the UI and docs.

**Project boundaries (in scope):**
- Public literature/guideline ingestion, entity/relation extraction, Neo4j graph construction
- Vector RAG baseline, GraphRAG pipeline, hybrid retrieval
- LLM-grounded generation with citations
- Benchmark-based evaluation comparing both pipelines
- React frontend for query, evidence, and graph visualization

**Out of scope:**
- Real patient data or EHR integration
- Any autonomous diagnostic or prescriptive capability
- Training a foundation model or biomedical LLM from scratch
- Production-grade auth, multi-tenant deployment, HIPAA compliance
- Ontology-scale graphs (SNOMED/UMLS-scale) — schema is deliberately small and hand-scoped

**Assumptions:**
- Access to an LLM API (or local LLM) for extraction assistance and generation
- Public/open-access clinical text sources are sufficient for a meaningful graph (not clinical-grade completeness)
- A small, curated benchmark (~50–80 questions) is sufficient for statistically indicative (not definitive) comparison

**Limitations (state openly in report):**
- Graph coverage is bounded by manual/semi-automated curation time, not exhaustive
- Extraction (NLP/LLM-assisted) will contain some error; validation step mitigates but doesn't eliminate this
- Results are indicative for the curated benchmark, not generalizable claims about GraphRAG vs RAG in general

**Research objective:**
Determine, using a controlled benchmark, whether graph-based retrieval measurably improves multi-hop clinical question answering versus a strong vector RAG baseline, and characterize *which query types* benefit.

**Expected contribution:**
- **Application contribution:** a working, explainable retrieval system with source-traceable answers
- **Research contribution:** an empirical, benchmark-based comparison of vector RAG vs GraphRAG on a hand-built clinical knowledge graph, with category-level breakdown (which question types benefit and by how much)
- **Engineering contribution:** a reusable ingestion → extraction → graph → hybrid-retrieval → evaluation pipeline architecture

---

## SECTION 2 — RESEARCH GAP

**Limitations of conventional RAG:**
Vector RAG retrieves the top-K chunks most similar to the query embedding. It has no explicit model of *relationships* between entities — it can only retrieve a chunk that happens to mention both entities together in the same passage. If the connecting fact requires combining two separate chunks (or documents), similarity search alone frequently fails to retrieve both, and even when it does, the LLM must infer the connection unaided and ungrounded.

**Why semantic similarity fails for relationship-heavy questions:**
Semantic similarity clusters text that reads similarly, not text that is *logically connected*. "Drug A treats Condition X" and "Condition X is common in Population Y" may be semantically distant (different vocabulary, different document sections) even though they combine into a valid multi-hop inference.

**Multi-hop reasoning, defined:**
Answering a question requires traversing more than one relationship edge — e.g., Drug → interacts with → Drug → contraindicated for → Condition — rather than reading a single fact in a single passage.

**Why clinical information is naturally relational:**
Drugs relate to conditions, symptoms, interactions, contraindications, populations, and evidence sources — a dense relational web that guidelines describe piecemeal across many documents and sections rather than as a single connected statement.

**Limitations of flat chunk retrieval:**
Chunking destroys document structure and cross-references; a chunk boundary can literally split a fact from its qualifier (e.g., "except in patients with renal impairment" landing in the next chunk).

**How knowledge graphs address this:**
A graph stores entities and relationships explicitly and persistently, independent of how they were originally phrased or which document they came from. Traversal can then follow explicit edges rather than relying on the LLM to notice an implicit connection.

**What GraphRAG adds:**
Query-time graph traversal (entity linking → multi-hop path retrieval → evidence collection) gives the LLM an explicit, structured evidence set for relational questions, rather than a similarity-ranked bag of text.

**What is NOT novel about this project (be honest):**
- GraphRAG as a general technique is established (Microsoft GraphRAG, various 2023–2025 papers). This project does not invent GraphRAG.
- Using Neo4j for knowledge graphs is standard industry practice.
- LLM-assisted entity/relation extraction is a known technique, not a new method.

**What the actual novelty/contribution should be:**
- A hand-scoped, small, high-quality clinical knowledge graph built specifically for this benchmark (not reusing an existing large ontology)
- A controlled, same-data, same-LLM, apples-to-apples empirical comparison between vector RAG and GraphRAG on a custom clinical multi-hop benchmark
- A category-level analysis identifying *which* clinical question types benefit most from graph retrieval — this is the genuine research finding, not "GraphRAG is better," but "GraphRAG improves X-hop and interaction-type questions by Y, with no significant difference on direct factual questions."

---

## SECTION 3 — RESEARCH QUESTION AND HYPOTHESIS

**Primary research question:**
Does graph-based (GraphRAG) retrieval improve multi-hop clinical knowledge question answering compared with conventional vector-based RAG, when both use the same source corpus and the same LLM?

**Secondary research questions:**
1. Which question categories (direct, relational, two-hop, multi-hop, interaction, contraindication) show the largest performance gap between the two methods?
2. Does hybrid retrieval (vector + graph) outperform either method alone?
3. What is the latency/quality tradeoff of graph traversal vs vector search?
4. Does increasing hop depth improve recall, and at what point does it introduce noise/irrelevant evidence?

**Hypothesis (H1):** GraphRAG will outperform vector RAG on two-hop, multi-hop, interaction, and contraindication question categories, measured by retrieval Precision@K/Recall@K and answer faithfulness.

**Null hypothesis (H0):** There is no statistically/practically meaningful difference in retrieval or answer quality between GraphRAG and vector RAG across question categories.

**Independent variable:** Retrieval method (Vector RAG / GraphRAG / Hybrid)

**Dependent variables:** Precision@K, Recall@K, answer faithfulness score, hallucination rate, citation correctness, latency

**Controlled/held constant:** Source corpus, chunking (where applicable), embedding model, LLM model and prompt template, benchmark question set

**Control/baseline:** Vector RAG is the baseline condition against which GraphRAG and Hybrid are compared.

**Experimental conditions:** (1) Vector-only, (2) Graph-only, (3) Hybrid — each run against the full benchmark set, same LLM, same evaluation rubric.

**Expected differential benefit:** Two-hop, multi-hop, interaction, and contraindication questions should benefit most from GraphRAG; direct factual and simple semantic questions should show little/no difference (both methods should handle these adequately) — this expected asymmetry is itself part of the hypothesis and should be explicitly tested, not assumed.

---

## SECTION 4 — USE CASES

All questions are framed as **knowledge retrieval from literature**, never as advice to a patient. Every UI answer must carry a visible disclaimer (see Section 23).

| # | Type | Example Question | Expected Entities | Expected Relationship(s) | Expected Retrieval Path | Expected Evidence | Expected Behavior | Failure Condition |
|---|------|------|------|------|------|------|------|------|
| 1 | Direct factual | "What is [Drug A] used to treat?" | Drug A | TREATS | Direct node lookup, 1 edge | Guideline stating indication | Answer with citation | Answers with no citation |
| 2 | Relationship | "What conditions is [Drug A] contraindicated for?" | Drug A | CONTRAINDICATED_FOR | 1-hop traversal | Contraindication list from guideline | Lists conditions with source | Misses a documented contraindication |
| 3 | Two-hop | "What side effects are associated with drugs that treat [Condition X]?" | Condition X, Drug(s) | TREATS → CAUSES(SideEffect) | 2-hop traversal (Condition→Drug→SideEffect) | Chained evidence from 2 documents | Answer synthesizes both hops, cites both | Only returns 1-hop info, misses side effects |
| 4 | Multi-hop | "Is there a documented interaction risk between drugs used for [Condition X] and drugs used for [Condition Y]?" | Condition X, Condition Y, Drug(s) | TREATS(x2) → INTERACTS_WITH | 3+ hop traversal across 2 branches | Cross-referenced interaction evidence | Correctly identifies indirect interaction path or states none found | Fabricates an interaction not in the graph |
| 5 | Interaction-related | "Does [Drug A] interact with [Drug B]?" | Drug A, Drug B | INTERACTS_WITH | Direct edge or "no edge found" | Interaction study/guideline entry | Explicit yes/no/insufficient-evidence with citation | States "no interaction" when evidence is simply absent (should say "no evidence found," not "safe") |
| 6 | Contraindication | "Is [Drug A] contraindicated in [Population/Condition]?" | Drug A, Population/Condition | CONTRAINDICATED_FOR | 1–2 hop | Guideline contraindication section | Direct answer + citation | Conflates "not recommended" with "contraindicated" |
| 7 | Evidence/citation | "What is the source for the claim that [Drug A] treats [Condition X]?" | Drug A, Condition X | TREATS (with provenance) | Node/edge → source document lookup | Full citation metadata | Returns document title, source, section, link | Cannot trace edge back to source (provenance gap) |
| 8 | Unanswerable | "What dosage of [Drug A] should a specific named patient take?" | Drug A | — | No valid graph/vector path for personalized dosing | None (out of scope) | Explicitly declines: "insufficient evidence / not a diagnostic tool," redirects to a professional | Fabricates a dosage or gives dosing advice |

**Design rule applied to every use case:** if evidence is absent, the system must say "no evidence found in the corpus" — never interpret absence as safety, and never interpret absence as license to guess.
