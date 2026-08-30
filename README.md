# Clinical GraphRAG Knowledge Retrieval & Decision Support System

## ⚠️ Safety Disclaimer & Limitations
*   **Research Prototype Only:** This is a clinical research prototype. It is **not** a medical device, is **not** cleared for diagnostic or treatment use, and must **not** be used to guide individual patient clinical care.
*   **No Claim of Clinical Effectiveness:** No claim of clinical efficacy, medical accuracy, or readiness for deployment in live healthcare workflows is made.
*   **Incomplete Context:** Clinical entity and relation extraction are automated and subject to parsing errors. The knowledge graph coverage is limited to the configured dataset and is **not** a comprehensive reference.

---

## 1. Project Overview & Architecture
This repository implements a **Clinical GraphRAG** prototype. The system compares traditional Vector RAG baseline retrieval against structured GraphRAG (traversing clinical entities and relationships via Neo4j) and Hybrid RAG (fusing semantic text search with path traversals) for multi-hop question answering.

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  React UI   │─────▶│  FastAPI     │─────▶│  Retrieval      │
│ (Vite+TS+   │◀─────│  Backend     │◀─────│  Layer          │
│  Tailwind)  │      │              │      │ (vector/graph/  │
│             │      │              │      │  hybrid)        │
└─────────────┘      └──────┬───────┘      └────────┬────────┘
                            │                        │
                   ┌────────▼────────┐      ┌────────▼────────┐
                   │  LLM Generation │      │  Neo4j Graph DB │
                   │  (API/local LLM)│      │  Vector Store   │
                   └─────────────────┘      │  (FAISS/Qdrant) │
                                            └─────────────────┘
```

---

## 2. Dataset Properties
The retrieval engine operates on a processed clinical literature dataset containing:
*   **Source Documents:** 20 openFDA drug label and clinical guideline source documents.
*   **Text Chunks:** 134 vector-indexed chunks.
*   **Validated Entities:** 290 graph nodes.
*   **Validated Relationships:** 247 graph edges.
*   **Coverage:** Note that entity/relationship extraction coverage is partial and limited to the sample documents.

---

## 3. Major Technologies
*   **Backend:** Python, FastAPI, Neo4j (AuraDB), FAISS Vector Store, Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Frontend:** React, TypeScript, Vite, Tailwind CSS, Cytoscape.js (Graph Visualization)

---

## 4. API Endpoints
*   `GET /`: Serving backend static root.
*   `GET /api/health`: Database and Vector store connectivity health status.
*   `POST /api/query`: Retrieve context and generate answers using `vector`, `graph`, or `hybrid` mode.
*   `GET /api/graph/neighborhood`: Fetch neighbors of an entity node for visual traversal.
*   `GET /api/graph/stats`: Fetch node and edge type summary statistics.
*   `GET /api/evidence/{source_id}`: Fetch raw clinical source text chunk metadata.

---

## 5. Local Setup & Environment Variables
Copy `.env.example` to `.env` in the root directory:
```bash
# Neo4j AuraDB Configuration
NEO4J_URI=neo4j+s://<your-auradb-subdomain>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-auradb-password>

# LLM Config (Leave blank or set to 'mock_key' for local mock generation)
LLM_API_KEY=mock_key
LLM_MODEL_NAME=gemini-3.5-flash

# Embeddings & Vector Settings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=./data/vector_index
```
> **Neo4j AuraDB Note:** The project requires connection to a live Neo4j AuraDB database pre-populated with clinical entities and relationships using the credentials specified above.

### Running Backend
```bash
# Set PYTHONPATH to project root
$env:PYTHONPATH="."
.venv/Scripts/python.exe backend/main.py
```
Backend runs at `http://localhost:8000`.

### Running Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173`.

---

## 6. Evaluation Summaries

### Retrieval Performance Summary (K=5)

| Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vector RAG** | **0.4892** | 0.6500 | 0.6500 | **28.54 ms** | **506.16 ms** |
| **GraphRAG** | 0.4375 | **0.8500** | **0.6667** | 711.52 ms | 1899.08 ms |
| **Hybrid RAG** | 0.2450 | 0.7500 | 0.5750 | 664.87 ms | 1578.17 ms |

* **Path Recovery Rate (q5-q9):** **GraphRAG and Hybrid RAG achieved 100.0% path recovery** (5 out of 5 exact multi-hop paths successfully reconstructed).

### Answer-Quality Performance Summary

| Metric | Vector RAG | GraphRAG | Hybrid RAG |
| :--- | :---: | :---: | :---: |
| **Mean Citation Precision** | **0.4892** | 0.4708 | 0.2421 |
| **Mean Citation Recall** | 0.6500 | **0.8500** | 0.7500 |
| **Mean Citation F1-Score** | 0.5333 | **0.5750** | 0.3592 |
| **Unsupported Claim Count** | **0** | **0** | **0** |
| **Unanswerable Refusal Rate** | 0.0% | **40.0%** (2/5) | 0.0% |

---

## 7. Production Deployment Guide

### Neo4j AuraDB Setup
Ensure your database instance is active in the cloud and configure the URI (`bolt+s://` or `neo4j+s://`), Username, and Password in the environment variables.

### Render Backend Deployment
1. Create a new **Web Service** on Render and connect your GitHub repository.
2. In the service settings:
   - **Environment:** `Docker`
   - **Docker Build Context:** `.` (repository root)
   - **Dockerfile Path:** `backend/Dockerfile`
3. Under **Environment Variables**, define:
   - `NEO4J_URI`
   - `NEO4J_USERNAME`
   - `NEO4J_PASSWORD`
   - `LLM_API_KEY` (or leave empty/set to `mock_key` for mock mode)
   - `FRONTEND_ORIGIN` (set to your Vercel frontend URL, e.g., `https://clinical-graphrag.vercel.app`)

### Vercel Frontend Deployment
1. Create a new project on Vercel and connect your repository.
2. In the project settings, configure:
   - **Framework Preset:** `Vite`
   - **Root Directory:** `frontend`
3. Under **Environment Variables**, define:
   - `VITE_API_BASE_URL` (set to your Render backend web service URL, e.g., `https://clinical-graphrag-backend.onrender.com`)
4. Click **Deploy**. Vercel will automatically build the static bundle and serve it.
