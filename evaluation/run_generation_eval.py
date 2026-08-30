"""
Generation Evaluation Runner.
Executes the answer generation for all 20 benchmark questions across 3 modes,
calculates objective answer-quality metrics, and outputs results.
"""
import os
import json
import time
import logging
import numpy as np
import re
from typing import List, Dict, Any

from backend.app.retrieval.vector_retrieve import vector_retrieve
from backend.app.retrieval.graph_retrieve import graph_retrieve
from backend.app.retrieval.hybrid_retrieve import hybrid_retrieve
from backend.app.generation.answer_generator import generate_answer, SAFETY_FOOTER
from backend.app.models.schemas import EvidenceItem
from backend.app.config import settings

# Force Mock LLM Mode to prevent calling Gemini API and ensure offline, zero-cost execution
settings.LLM_API_KEY = ""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag.evaluation.generation")

def calculate_citation_metrics(generated_citations: List[str], expected_source_ids: List[str]) -> Dict[str, float]:
    if not expected_source_ids:
        # Unanswerable: expect no citations
        return {
            "precision": 1.0 if not generated_citations else 0.0,
            "recall": 1.0 if not generated_citations else 0.0,
            "f1": 1.0 if not generated_citations else 0.0
        }
    
    if not generated_citations:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        
    expected_set = set(expected_source_ids)
    hits = sum(1 for c in generated_citations if c in expected_set)
    
    precision = hits / len(generated_citations)
    recall = hits / len(expected_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def run_generation_evaluation():
    benchmark_path = "evaluation/benchmark/benchmark_set.json"
    results_dir = "evaluation/results"
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    items = benchmark_data["benchmark_items"]
    records = []

    os.makedirs(results_dir, exist_ok=True)

    for item in items:
        q_id = item["id"]
        question = item["question"]
        category = item["category"]
        expected_ids = item["expected_source_ids"]
        answerable = item["answerable"]

        for mode in ["vector", "graph", "hybrid"]:
            t0 = time.perf_counter()
            evidence_items = []
            graph_paths = []
            status = "ok"

            try:
                # Retrieve evidence
                if mode == "vector":
                    evidence_items = vector_retrieve(question, top_k=5)
                elif mode == "graph":
                    g_res = graph_retrieve(question)
                    evidence_items = g_res.get("evidence_items", [])
                    graph_paths = g_res.get("graph_paths", [])
                    status = g_res.get("status", "ok")
                else:  # hybrid
                    h_res = hybrid_retrieve(question, top_k=5)
                    evidence_items = h_res.get("evidence_items", [])
                    graph_paths = h_res.get("graph_paths", [])
                    status = h_res.get("status", "ok")

                # Generate Answer
                answer_res = generate_answer(question, evidence_items, mode)
                
                # Check refusal behavior
                answer_text = answer_res.answer_text
                is_refusal = answer_text.startswith("I do not have sufficient evidence") or "personalized clinical advice" in answer_text
                
                # Citation analysis
                citations = [c.source_id for c in answer_res.citations]
                
                # Compute metrics
                cit_metrics = calculate_citation_metrics(citations, expected_ids)
                
                # Check for unsupported claims: no citations provided when text is present and not a refusal
                has_text = len(answer_text.replace(SAFETY_FOOTER, "").strip()) > 0
                unsupported_claims = has_text and not is_refusal and not citations

                status_code = "success"
            except Exception as e:
                logger.error(f"Failed generation for {q_id} in {mode} mode: {str(e)}")
                answer_text = f"Error: {str(e)}"
                citations = []
                cit_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                is_refusal = False
                unsupported_claims = False
                status_code = f"failed: {str(e)}"

            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000.0

            records.append({
                "question_id": q_id,
                "category": category,
                "answerable": answerable,
                "retrieval_mode": mode,
                "expected_source_ids": expected_ids,
                "generated_answer_text": answer_text,
                "generated_citations": citations,
                "citation_precision": cit_metrics["precision"],
                "citation_recall": cit_metrics["recall"],
                "citation_f1": cit_metrics["f1"],
                "is_refusal": is_refusal,
                "unsupported_claims": unsupported_claims,
                "latency_ms": latency,
                "status": status_code
            })

    # Save generation_results.json
    results_path = os.path.join(results_dir, "generation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Compute aggregate summary
    summary = compute_generation_summary(records)
    summary_path = os.path.join(results_dir, "generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Generation quality evaluation completed successfully.")

def compute_generation_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def agg_subset(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        precisions = [r["citation_precision"] for r in subset]
        recalls = [r["citation_recall"] for r in subset]
        f1s = [r["citation_f1"] for r in subset]
        latencies = [r["latency_ms"] for r in subset]
        
        unsupported_count = sum(1 for r in subset if r["unsupported_claims"])
        
        unans_subset = [r for r in subset if not r["answerable"]]
        total_unans = len(unans_subset)
        refused_unans = sum(1 for r in unans_subset if r["is_refusal"])
        refusal_rate = (refused_unans / total_unans) if total_unans > 0 else 1.0

        return {
            "mean_citation_precision": float(np.mean(precisions)) if precisions else 0.0,
            "mean_citation_recall": float(np.mean(recalls)) if recalls else 0.0,
            "mean_citation_f1": float(np.mean(f1s)) if f1s else 0.0,
            "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "unsupported_claim_count": unsupported_count,
            "unanswerable_refusal_rate": refusal_rate
        }

    by_mode = {}
    for mode in ["vector", "graph", "hybrid"]:
        sub = [r for r in records if r["retrieval_mode"] == mode]
        by_mode[mode] = agg_subset(sub)

    # By category
    by_category = {}
    cats = sorted(list(set(r["category"] for r in records)))
    for cat in cats:
        by_category[cat] = {}
        for mode in ["vector", "graph", "hybrid"]:
            sub = [r for r in records if r["category"] == cat and r["retrieval_mode"] == mode]
            by_category[cat][mode] = agg_subset(sub)

    return {
        "by_retrieval_mode": by_mode,
        "by_category": by_category
    }

if __name__ == "__main__":
    run_generation_evaluation()
