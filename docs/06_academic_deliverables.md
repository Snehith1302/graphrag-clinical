# 06 — Academic Deliverables: README, Paper, Report, Viva, Interview Prep

## SECTION 33 — README SPECIFICATION (GitHub)

```markdown
# GraphRAG-Based Clinical Knowledge Retrieval and Decision Support System

## ⚠️ Safety Disclaimer
Research prototype only. Not a medical device. Does not diagnose or treat.
Not for real clinical decisions. See LIMITATIONS.

## Overview
[100-word overview from Section 1]

## Architecture
[ASCII diagram from Section 22 + short explanation]

## Features
- Vector RAG baseline, GraphRAG, and Hybrid retrieval modes
- Neo4j clinical knowledge graph with full source provenance
- Explainable "evidence trace" / retrieval-path visualization
- Benchmark-based evaluation comparing all three retrieval modes

## Tech Stack
Backend: Python, FastAPI, Neo4j, sentence-transformers, FAISS/Qdrant
Frontend: React, TypeScript, Vite, Tailwind CSS, Cytoscape.js/React Flow

## Setup
1. Clone repo
2. Copy `.env.example` → `.env`, fill in values
3. `docker-compose up`
4. Run ingestion: `python ingestion/run_ingestion.py`
5. Open frontend at `localhost:5173`

## Environment Variables
[table matching Section 43]

## Data Pipeline
[summary of Section 8]

## Graph Schema
[summary table from Section 6, link to full docs/02_data_graph_schema.md]

## Retrieval Pipeline
[summary of Sections 11-13]

## Evaluation
[summary of results once available — table of Vector vs Graph vs Hybrid metrics]

## Screenshots
[placeholders]

## Limitations
[from Section 1 — extraction noise, bounded corpus, indicative not definitive results]

## Future Work
[from Section 38]
```

## SECTION 34 — RESEARCH PAPER STRUCTURE

1. **Abstract** — problem, method, key finding (numbers filled in only after experiments)
2. **Introduction** — motivation, contribution summary
3. **Problem Statement** — vector RAG's multi-hop limitation
4. **Related Work** — cite GraphRAG literature, vector RAG literature, biomedical NER work (real citations only — no fabricated references)
5. **Research Gap** — Section 2 content, condensed
6. **Methodology** — ingestion, extraction, graph construction, retrieval pipelines
7. **Architecture** — system diagrams
8. **Dataset** — sources, scope, licensing
9. **Knowledge Graph Construction** — schema, extraction method, validation
10. **Retrieval Method** — vector/graph/hybrid details
11. **Experimental Setup** — benchmark design, metrics, conditions
12. **Results** — real numbers, tables, category breakdown
13. **Discussion** — what the results mean, which hypothesis is supported
14. **Limitations** — corpus size, extraction noise, indicative not definitive
15. **Conclusion** — restate finding
16. **Future Work** — Section 38
17. **References** — only real, verifiable sources

## SECTION 35 — FINAL-YEAR PROJECT REPORT STRUCTURE (University format)

1. Certificate / front matter placeholders
2. Abstract
3. Introduction
4. Literature Survey (vector RAG, knowledge graphs, GraphRAG, biomedical NLP)
5. Existing System (standard RAG chatbots — limitations)
6. Proposed System (GraphRAG comparison approach)
7. Requirements (functional/non-functional, hardware/software)
8. Methodology (pipeline stages, Sections 8-15 condensed)
9. Architecture (Section 22 diagrams)
10. Implementation (module-by-module walkthrough, code structure from Section 21)
11. Results (evaluation tables/charts from real experiments)
12. Testing (Section 31 summary + results)
13. Conclusion
14. Future Work
15. References

## SECTION 36 — VIVA PREPARATION (selected — representative set across all categories)

| Category | Question | Concise Answer | Deeper Answer | Keywords |
|---|---|---|---|---|
| A. Basic | What is RAG? | Retrieval-Augmented Generation: retrieve relevant context, then generate an answer grounded in it. | Combines a retriever (search over a corpus) with a generator (LLM) so answers are grounded in retrieved evidence rather than purely parametric knowledge, reducing hallucination. | retrieval, generation, grounding |
| A. Basic | Why not just use a bigger LLM with no retrieval? | LLMs' internal knowledge is static, unverifiable, and can't cite sources. | Retrieval adds up-to-date, source-traceable, domain-specific knowledge without retraining the model, and enables citation-based verification. | parametric vs non-parametric knowledge |
| B. RAG | What is chunking and why does it matter? | Splitting documents into retrievable pieces. | Chunk size/overlap affects retrieval precision (too big = noisy context) and recall (too small = lost context); a core RAG design tradeoff. | chunk size, overlap, context window |
| B. RAG | What's a limitation of vector RAG this project addresses? | It can't explicitly connect facts across multiple chunks/documents (multi-hop). | Similarity search retrieves by surface semantic closeness, not logical connection, so relationship-heavy questions requiring 2+ hops often fail. | multi-hop, semantic similarity |
| C. Knowledge Graphs | Why a graph instead of a table for this data? | Graphs naturally represent many-to-many relationships and variable-depth traversal. | Relational databases require pre-defined joins; graphs allow flexible, arbitrary-depth traversal (e.g., 3-hop path) without schema changes, matching the exploratory nature of clinical relationships. | nodes, edges, traversal |
| D. Neo4j | Why Neo4j specifically? | Native graph database with Cypher, mature ecosystem, free community edition. | Purpose-built index-free adjacency makes traversal queries fast regardless of graph size, unlike simulating graphs in SQL with recursive joins. | index-free adjacency, Cypher |
| E. Cypher | How do you prevent Cypher injection? | Parameterized queries only, never string concatenation. | The Neo4j driver binds parameters separately from the query text, so user input can never alter query structure — identical principle to SQL prepared statements. | parameter binding, injection |
| F. NLP | Why hybrid NER (spaCy + LLM) instead of pure LLM extraction? | Balances cost and schema conformance. | spaCy/scispaCy narrows candidate spans cheaply; LLM only classifies/normalizes those spans into the closed schema, reducing both API cost and free-text drift from the fixed entity type list. | NER, candidate spans, schema conformance |
| G. LLMs | How do you stop the LLM from hallucinating? | Strict prompt: answer only from evidence, cite everything, say "insufficient evidence" when unsure. | Combined with a closed evidence set (not open web), explicit uncertainty instructions, and a hallucination-rate metric in evaluation, hallucination is both discouraged by prompt design and measured, not just assumed away. | grounding, faithfulness, hallucination rate |
| H. GraphRAG | What does GraphRAG add over vanilla RAG? | Explicit graph traversal to find multi-hop evidence paths, not just similarity-ranked chunks. | Entity linking + bounded-depth traversal over pre-extracted, validated relationships gives structured, explainable evidence for relational questions. | entity linking, traversal, evidence path |
| I. Multi-hop | What's your max hop depth and why? | 3, to bound noise and combinatorial explosion. | Beyond 3 hops, path relevance drops sharply and the number of possible paths grows exponentially, degrading both precision and latency — empirically justified via ablation, not arbitrary. | hop depth, combinatorial explosion |
| J. Evaluation | How do you measure faithfulness? | Check if every answer claim traces to cited evidence. | Combination of manual rubric scoring (primary) and LLM-as-judge cross-check (secondary, flagged as a proxy since it risks circularity if the same model both generates and judges). | faithfulness, LLM-as-judge, circularity |
| K. System Design | Why keep vector RAG as a baseline rather than only building GraphRAG? | Without a fair baseline there's no valid comparison — the whole research question depends on it. | A weak/strawman baseline would invalidate any claimed improvement; the baseline must use the same corpus, same LLM, same prompt to isolate retrieval method as the only variable. | controlled comparison, baseline validity |
| L. Security | How is patient privacy handled? | No real patient data is used anywhere in the system. | The dataset is entirely public literature/guidelines; there's no personal data to protect at the input level, though the citation-provenance requirement still applies to protect source attribution. | no PHI, public corpus |
| M. Healthcare Limitations | Could this be used clinically today? | No — it's an explicitly non-deployable research prototype. | Corpus coverage, extraction accuracy, and evaluation scale are all far below what real clinical decision support would require; the disclaimer and scope boundary are load-bearing, not decorative. | prototype, non-deployable, scope |
| N. Research Methodology | Why is this a valid research contribution if GraphRAG already exists? | The contribution is the controlled empirical comparison on a custom benchmark, not the invention of GraphRAG. | Novelty is in generating category-level empirical evidence (which question types benefit, by how much) on a purpose-built clinical graph — a legitimate, scoped undergraduate research contribution. | empirical comparison, category-level findings |

*(This table is illustrative of the required depth/format across all 14 categories — extend using the same concise/deep/keyword structure to reach 50 total once your actual implementation choices are finalized, since some answers will depend on decisions made during Phases 1–3, e.g., exact embedding model chosen.)*

## SECTION 37 — INTERVIEW EXPLANATION

**30-second:** "I built a system comparing two ways of doing RAG on clinical literature — standard vector search versus a Neo4j knowledge graph — to test whether explicit graph traversal improves multi-hop question answering, like connecting a drug's interactions across multiple documents."

**1-minute:** adds: "I built an ingestion pipeline that extracts entities and relationships from public clinical guidelines and drug labels using a hybrid spaCy+LLM approach, populated a Neo4j graph with full source provenance, then built three retrieval modes — vector, graph, and hybrid — behind the same LLM and prompt so the comparison is fair. I evaluated all three on a hand-built benchmark across categories like direct facts, two-hop, and drug interactions."

**3-minute technical:** adds architecture flow, schema design choices (closed relation-type set for Cypher-injection safety), evaluation metrics (precision/recall/faithfulness/hallucination rate), and the honest limitation that results are indicative on a bounded corpus, not a generalizable clinical claim.

**Resume bullets:**
- Built a GraphRAG system on Neo4j comparing graph-based vs vector-based retrieval for multi-hop clinical QA, with a custom benchmark showing category-level performance differences
- Designed a closed-schema entity/relation extraction pipeline (hybrid NER + LLM) with full source provenance and injection-safe parameterized Cypher retrieval
- Implemented and evaluated 3 retrieval architectures (vector RAG, GraphRAG, hybrid) under a controlled, same-LLM comparison methodology

**LinkedIn:** "Researched and built a GraphRAG-based clinical knowledge retrieval system, empirically comparing it against standard vector RAG on a custom benchmark of multi-hop clinical questions — with full source provenance, an explainable evidence-trace UI, and a closed, injection-safe graph query layer."

**GitHub description:** "GraphRAG vs vector RAG on clinical literature — Neo4j knowledge graph, FastAPI backend, React frontend, benchmark-based evaluation. Research prototype, not a medical tool."

## SECTION 38 — FUTURE WORK

**Realistic near-term future work:**
- Larger/biomedical-specific NER model integration for better extraction recall
- Expanded benchmark set with more categories/questions for stronger statistical power
- Cross-encoder reranking as a standard (not optional) hybrid step
- Contradiction detection surfaced more systematically (currently just flagged, not resolved)

**Advanced research possibilities (explicitly beyond this project's scope):**
- Ontology integration (SNOMED CT/UMLS-scale) for broader coverage
- Temporal reasoning (guideline changes over time, superseded recommendations)
- Multimodal evidence (tables, dosage charts as structured evidence, not just text)
- Personalized retrieval (would require real clinical data + ethics approval — explicitly out of this project's reach)
- Human clinician evaluation study (would require IRB-level review, out of undergraduate scope)

## SECTION 39 — RISKS AND MITIGATION

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Dataset quality/coverage too thin | Medium | High | Bound scope per Section 5, prioritize openFDA (pre-structured) as anchor source |
| Extraction errors (entity/relation) | High | Medium | Validation stage + manual spot-check sample + confidence thresholds |
| Graph sparsity limits multi-hop benchmark questions | Medium | High | Design benchmark questions *after* seeing what the graph actually contains, not before |
| Hallucination in generation | Medium | High | Strict grounding prompt + hallucination-rate metric + "insufficient evidence" fallback |
| Insufficient evidence for some benchmark questions | Medium | Low | This is a valid benchmark category (Use Case 8) — not a failure, an expected test case |
| Inconsistent terminology across sources | Medium | Medium | Normalization/synonym map (Section 9), manual review pass |
| LLM API costs | Low–Medium | Medium | Bound extraction to pre-detected spans only (not full-document LLM calls), cap benchmark size |
| Performance/latency at demo time | Low | Medium | Bounded hop depth, local Neo4j instance, small corpus keeps queries fast |
| Project scope creep (over-engineering) | High | High | Hold firm to Section 1's explicit out-of-scope list; resist adding node types/features not justified by actual extracted data |
| Evaluation difficulty (subjective scoring) | Medium | Medium | Dual scoring (manual rubric + LLM-as-judge) with manual as primary source of truth |
