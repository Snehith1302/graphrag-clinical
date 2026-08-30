# 02 — Dataset Strategy, Knowledge Graph Schema, Graph Examples

## SECTION 5 — DATASET STRATEGY

**No real patient data. Ever.** All sources below are public, literature/guideline-level, not individual records.

| Source | Contains | Access/License | Suitability | Preprocessing needed |
|---|---|---|---|---|
| PubMed/PMC open-access subset | Abstracts + full text of biomedical research articles | Free API (NCBI E-utilities); open-access subset explicitly licensed for reuse | High — rich relational content (drug studies, interactions) | XML/text extraction, reference stripping, section splitting |
| Public clinical guideline PDFs (e.g., WHO, NICE, openly published national guidelines) | Structured treatment/contraindication guidance | Varies — check each publisher's reuse terms; prefer explicitly open ones | High — dense TREATS/CONTRAINDICATED_FOR relations, well-structured | PDF parsing, table extraction, section-heading normalization |
| DrugBank open/academic subset (if license permits for your institution) | Drug–drug interactions, indications, mechanisms | Free academic license available (check terms before using) | High if accessible; if not, skip and rely on guidelines/PubMed | Structured data → easier extraction |
| Open FDA drug label data (openFDA API) | Structured drug label sections (indications, interactions, contraindications) | Public domain (US government data) | High — pre-structured, low extraction noise, safe reuse | JSON parsing, section mapping to schema |
| Wikipedia/Wikidata (medical subset) | General drug/condition facts, sometimes structured relations | CC-BY-SA — must attribute, reuse permitted | Medium — useful for bootstrapping entity lists, not primary evidence source | Entity list extraction only, not primary evidence |

**Recommended primary sources for a bounded 10-week build:** openFDA drug labels (best signal-to-effort ratio — already semi-structured) + a small curated set of open-access PubMed/PMC articles + 1–2 open clinical guideline documents for contraindication-style relations.

**Dataset scope:**
- Minimum: 15–20 source documents, ~150 unique entities, ~300 unique relationships (enough to demonstrate multi-hop paths exist)
- Recommended: 30–50 source documents, ~300–500 entities, ~600–1000 relationships
- Maximum reasonable scope for 10 weeks: ~80 documents — beyond this, extraction validation time dominates and risks schedule slip

**Document formats:** PDF (guidelines), JSON (openFDA), XML/text (PubMed/PMC)

**Metadata required per document:** document ID, title, source name, authors (if available), publication year, source URL (if legally reusable), retrieval date, document type (guideline / research article / drug label)

---

## SECTION 6 — KNOWLEDGE GRAPH SCHEMA

### Node Types

| Node Type | Purpose | Required Properties | Optional Properties | Unique ID |
|---|---|---|---|---|
| `Drug` | Represents a medication/substance | `name`, `normalized_name`, `id` | `drug_class`, `mechanism` | `drug_id` (slug of normalized_name) |
| `Condition` | Disease/medical condition | `name`, `normalized_name`, `id` | `icd_code` (if available) | `condition_id` |
| `Symptom` | Clinical symptom/sign | `name`, `id` | `description` | `symptom_id` |
| `SideEffect` | Adverse effect of a drug | `name`, `id`, `severity` (if stated) | `frequency` | `side_effect_id` |
| `Population` | Patient subgroup (e.g., "renal impairment", "pregnancy") | `name`, `id` | `description` | `population_id` |
| `ClinicalStudy` | A research article/study used as evidence | `title`, `id`, `year` | `authors`, `journal` | `study_id` |
| `Guideline` | A clinical guideline document | `title`, `id`, `publisher`, `year` | `url` | `guideline_id` |

*Kept deliberately small — no `Organization`, `Biomarker`, or `Treatment` node types unless the extracted data actually produces enough instances to justify them (avoid schema bloat / over-engineering per project principles).*

### Relationship Types

| Relationship | Source → Target | Meaning | Properties | Evidence requirement |
|---|---|---|---|---|
| `TREATS` | Drug → Condition | Drug is indicated for condition | `confidence`, `source_id` | Must link to ≥1 Guideline/Study/label |
| `CAUSES` | Drug → SideEffect | Drug causes this adverse effect | `confidence`, `frequency`, `source_id` | Must link to source |
| `HAS_SYMPTOM` | Condition → Symptom | Condition presents with this symptom | `confidence`, `source_id` | Must link to source |
| `INTERACTS_WITH` | Drug → Drug | Documented interaction between two drugs | `severity`, `confidence`, `source_id` | Must link to source (bidirectional edge or two directed edges) |
| `CONTRAINDICATED_FOR` | Drug → Condition \| Population | Drug should not be used given this condition/population | `confidence`, `source_id` | Must link to source |
| `RECOMMENDS` | Guideline → Drug | Guideline recommends this drug for a context | `context`, `source_id` | Self-referential (the guideline IS the source) |
| `EVIDENCED_BY` | Any relationship-bearing edge → ClinicalStudy/Guideline | Provenance link | `excerpt_span` | Always required — this is the provenance mechanism itself |

### Constraints & Uniqueness Rules (Cypher)

```cypher
CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_id IS UNIQUE;
CREATE CONSTRAINT condition_id_unique IF NOT EXISTS FOR (c:Condition) REQUIRE c.condition_id IS UNIQUE;
CREATE CONSTRAINT symptom_id_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.symptom_id IS UNIQUE;
CREATE CONSTRAINT sideeffect_id_unique IF NOT EXISTS FOR (se:SideEffect) REQUIRE se.side_effect_id IS UNIQUE;
CREATE CONSTRAINT population_id_unique IF NOT EXISTS FOR (p:Population) REQUIRE p.population_id IS UNIQUE;
CREATE CONSTRAINT study_id_unique IF NOT EXISTS FOR (st:ClinicalStudy) REQUIRE st.study_id IS UNIQUE;
CREATE CONSTRAINT guideline_id_unique IF NOT EXISTS FOR (g:Guideline) REQUIRE g.guideline_id IS UNIQUE;
```

Every relationship-creating Cypher statement must be paired with an `EVIDENCED_BY` edge to a source node — **no edge may be inserted without provenance.** This is a hard rule for the ingestion pipeline (see Section 8), not optional.

---

## SECTION 7 — KNOWLEDGE GRAPH EXAMPLES

Below: 10 example structures. A. conceptual text diagram, B. Cypher.

**1. Simple TREATS**
A: `(Drug:Metformin) -[TREATS]-> (Condition:Type2Diabetes)`
B:
```cypher
MERGE (d:Drug {drug_id:'metformin'}) SET d.name='Metformin'
MERGE (c:Condition {condition_id:'type2_diabetes'}) SET c.name='Type 2 Diabetes'
MERGE (d)-[r:TREATS {confidence:0.95, source_id:'guideline_001'}]->(c)
```

**2. Contraindication**
A: `(Drug:Metformin) -[CONTRAINDICATED_FOR]-> (Population:SevereRenalImpairment)`
B: `MERGE (d:Drug {drug_id:'metformin'})-[:CONTRAINDICATED_FOR {source_id:'label_003'}]->(p:Population {population_id:'severe_renal_impairment'})`

**3. Drug interaction**
A: `(Drug:WarfarinA) -[INTERACTS_WITH {severity:'high'}]-> (Drug:AspirinB)`
B: `MERGE (a:Drug{drug_id:'warfarin'})-[:INTERACTS_WITH {severity:'high', source_id:'study_012'}]->(b:Drug{drug_id:'aspirin'})`

**4. Two-hop chain (Condition → Drug → SideEffect)**
A: `(Condition:Hypertension) <-[TREATS]- (Drug:Lisinopril) -[CAUSES]-> (SideEffect:DryCough)`
B:
```cypher
MATCH (c:Condition{condition_id:'hypertension'})<-[:TREATS]-(d:Drug)-[:CAUSES]->(se:SideEffect)
RETURN d.name, se.name
```

**5. Symptom association**
A: `(Condition:Type2Diabetes) -[HAS_SYMPTOM]-> (Symptom:Polyuria)`
B: `MERGE (c:Condition{condition_id:'type2_diabetes'})-[:HAS_SYMPTOM {source_id:'study_004'}]->(s:Symptom{symptom_id:'polyuria'})`

**6. Guideline recommendation**
A: `(Guideline:ADAGuideline2024) -[RECOMMENDS]-> (Drug:Metformin)`
B: `MERGE (g:Guideline{guideline_id:'ada_2024'})-[:RECOMMENDS {context:'first-line therapy'}]->(d:Drug{drug_id:'metformin'})`

**7. Provenance edge**
A: `(TREATS edge Metformin→Type2Diabetes) -[EVIDENCED_BY]-> (Guideline:ADAGuideline2024)`
B: modeled via `source_id` property on the relationship, resolved at query time by joining to the Guideline/Study node with matching id (avoids Neo4j's edge-to-edge limitation).

**8. Multi-hop cross-condition interaction**
A: `(Condition:Depression) <-[TREATS]- (Drug:SSRI_X) -[INTERACTS_WITH]-> (Drug:NSAID_Y) -[TREATS]-> (Condition:Arthritis)`
B:
```cypher
MATCH path = (c1:Condition)<-[:TREATS]-(d1:Drug)-[:INTERACTS_WITH]-(d2:Drug)-[:TREATS]->(c2:Condition)
WHERE c1.condition_id = 'depression' AND c2.condition_id = 'arthritis'
RETURN path
```

**9. Population-qualified contraindication with evidence**
A: `(Drug:Ibuprofen) -[CONTRAINDICATED_FOR {source_id:'label_009'}]-> (Population:ThirdTrimesterPregnancy)`
B: `MERGE (d:Drug{drug_id:'ibuprofen'})-[:CONTRAINDICATED_FOR {source_id:'label_009'}]->(p:Population{population_id:'third_trimester_pregnancy'})`

**10. Side-effect shared by two drugs (used for "which drugs share this risk" queries)**
A: `(Drug:DrugA) -[CAUSES]-> (SideEffect:Nausea) <-[CAUSES]- (Drug:DrugB)`
B:
```cypher
MATCH (d1:Drug)-[:CAUSES]->(se:SideEffect{symptom_id:'nausea'})<-[:CAUSES]-(d2:Drug)
WHERE d1 <> d2
RETURN d1.name, d2.name
```

Each of these supports retrieval by giving GraphRAG an explicit path to traverse (vs. vector RAG, which would need the connecting fact to appear in a single retrieved chunk).
