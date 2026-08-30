"""
Evaluation Runner for clinical GraphRAG repository.
Executes benchmark questions against Vector, Graph, and Hybrid retrieval modes,
measures latency, calculates precision/recall/MRR metrics, and serializes results.
"""
import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from backend.app.retrieval.vector_retrieve import vector_retrieve
from backend.app.retrieval.graph_retrieve import graph_retrieve
from backend.app.retrieval.hybrid_retrieve import hybrid_retrieve

logger = logging.getLogger("graphrag.evaluation.runner")

def calculate_metrics(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, and MRR.
    """
    if not expected_ids:
        # For unanswerable questions
        return {
            f"precision_at_{k}": 1.0 if not retrieved_ids else 0.0,
            f"recall_at_{k}": 1.0 if not retrieved_ids else 0.0,
            "mrr": 1.0 if not retrieved_ids else 0.0
        }

    retrieved_k = retrieved_ids[:k]
    expected_set = set(expected_ids)
    
    # Precision@K
    hits = sum(1 for rid in retrieved_k if rid in expected_set)
    precision = hits / k
    
    # Recall@K
    recall = hits / len(expected_set)
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            mrr = 1.0 / (idx + 1)
            break
            
    return {
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        "mrr": mrr
    }

def run_evaluation(
    benchmark_path: str = "evaluation/benchmark/benchmark_set.json",
    results_dir: str = "evaluation/results"
) -> Dict[str, Any]:
    """
    Runs evaluation on the benchmark set using Vector, Graph, and Hybrid retrieval modes.
    """
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark set not found at {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    items = benchmark_data.get("benchmark_items", [])
    run_records = []
    
    os.makedirs(results_dir, exist_ok=True)

    for item in items:
        q_id = item["id"]
        question = item["question"]
        category = item["category"]
        expected_ids = item["expected_source_ids"]
        answerable = item["answerable"]
        
        # 1. Vector RAG Retrieval
        t0 = time.perf_counter()
        try:
            vector_items = vector_retrieve(question, top_k=5)
            vector_retrieved_ids = []
            for vi in vector_items:
                vector_retrieved_ids.extend(vi.source_ids)
            vector_retrieved_ids = list(dict.fromkeys(vector_retrieved_ids)) # deduplicate
            vector_status = "success"
        except Exception as e:
            logger.error(f"Vector retrieve failed for {q_id}: {str(e)}")
            vector_retrieved_ids = []
            vector_status = f"failed: {str(e)}"
        t1 = time.perf_counter()
        vector_latency = (t1 - t0) * 1000.0 # milliseconds
        vector_metrics = calculate_metrics(vector_retrieved_ids, expected_ids, k=5)
        
        # Unanswerable checks
        evidence_retrieved = len(vector_retrieved_ids) > 0
        insufficient_evidence_state = len(vector_retrieved_ids) == 0

        run_records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "vector",
            "retrieved_source_ids": vector_retrieved_ids,
            "expected_source_ids": expected_ids,
            "graph_paths": [],
            "number_of_retrieved_evidence_items": len(vector_retrieved_ids),
            "precision_at_5": vector_metrics["precision_at_5"],
            "recall_at_5": vector_metrics["recall_at_5"],
            "mrr": vector_metrics["mrr"],
            "retrieval_latency_ms": vector_latency,
            "total_latency_ms": vector_latency,
            "answer_text": "",
            "status": vector_status,
            "unanswerable_eval": {
                "evidence_retrieved": evidence_retrieved,
                "insufficient_evidence_state": insufficient_evidence_state
            }
        })

        # 2. Graph RAG Retrieval
        t0 = time.perf_counter()
        try:
            graph_res = graph_retrieve(question)
            graph_retrieved_ids = graph_res.get("source_ids", [])
            graph_paths = graph_res.get("graph_paths", [])
            graph_status = graph_res.get("status", "ok")
        except Exception as e:
            logger.error(f"Graph retrieve failed for {q_id}: {str(e)}")
            graph_retrieved_ids = []
            graph_paths = []
            graph_status = f"failed: {str(e)}"
        t1 = time.perf_counter()
        graph_latency = (t1 - t0) * 1000.0
        graph_metrics = calculate_metrics(graph_retrieved_ids, expected_ids, k=5)
        
        # Unanswerable checks
        evidence_retrieved = len(graph_retrieved_ids) > 0
        insufficient_evidence_state = (graph_status == "insufficient_evidence")

        run_records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "graph",
            "retrieved_source_ids": graph_retrieved_ids,
            "expected_source_ids": expected_ids,
            "graph_paths": graph_paths,
            "number_of_retrieved_evidence_items": len(graph_retrieved_ids),
            "precision_at_5": graph_metrics["precision_at_5"],
            "recall_at_5": graph_metrics["recall_at_5"],
            "mrr": graph_metrics["mrr"],
            "retrieval_latency_ms": graph_latency,
            "total_latency_ms": graph_latency,
            "answer_text": "",
            "status": graph_status,
            "unanswerable_eval": {
                "evidence_retrieved": evidence_retrieved,
                "insufficient_evidence_state": insufficient_evidence_state
            }
        })

        # 3. Hybrid RAG Retrieval
        t0 = time.perf_counter()
        try:
            hybrid_res = hybrid_retrieve(question, top_k=5)
            hybrid_retrieved_ids = hybrid_res.get("source_ids", [])
            hybrid_paths = hybrid_res.get("graph_paths", [])
            hybrid_status = hybrid_res.get("status", "ok")
        except Exception as e:
            logger.error(f"Hybrid retrieve failed for {q_id}: {str(e)}")
            hybrid_retrieved_ids = []
            hybrid_paths = []
            hybrid_status = f"failed: {str(e)}"
        t1 = time.perf_counter()
        hybrid_latency = (t1 - t0) * 1000.0
        hybrid_metrics = calculate_metrics(hybrid_retrieved_ids, expected_ids, k=5)
        
        # Unanswerable checks
        evidence_retrieved = len(hybrid_retrieved_ids) > 0
        insufficient_evidence_state = (hybrid_status == "insufficient_evidence")

        run_records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "hybrid",
            "retrieved_source_ids": hybrid_retrieved_ids,
            "expected_source_ids": expected_ids,
            "graph_paths": hybrid_paths,
            "number_of_retrieved_evidence_items": len(hybrid_retrieved_ids),
            "precision_at_5": hybrid_metrics["precision_at_5"],
            "recall_at_5": hybrid_metrics["recall_at_5"],
            "mrr": hybrid_metrics["mrr"],
            "retrieval_latency_ms": hybrid_latency,
            "total_latency_ms": hybrid_latency,
            "answer_text": "",
            "status": hybrid_status,
            "unanswerable_eval": {
                "evidence_retrieved": evidence_retrieved,
                "insufficient_evidence_state": insufficient_evidence_state
            }
        })

    # Save retrieval_results.json
    results_path = os.path.join(results_dir, "retrieval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(run_records, f, indent=2, ensure_ascii=False)

    # Compute summaries
    summary_report = compute_summary(run_records)
    summary_path = os.path.join(results_dir, "retrieval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    return {
        "results": run_records,
        "summary": summary_report
    }

def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregates overall, by category, and by retrieval mode.
    """
    def aggregate_group(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        precisions = [r["precision_at_5"] for r in subset]
        recalls = [r["recall_at_5"] for r in subset]
        mrrs = [r["mrr"] for r in subset]
        latencies = [r["retrieval_latency_ms"] for r in subset]
        
        # Unanswerable checks on subset
        unans_subset = [r for r in subset if not r["answerable"]]
        unans_count = len(unans_subset)
        returned_evidence_count = sum(1 for r in unans_subset if r["unanswerable_eval"]["evidence_retrieved"])
        insufficient_evidence_count = sum(1 for r in unans_subset if r["unanswerable_eval"]["insufficient_evidence_state"])

        return {
            "mean_precision_at_5": float(np.mean(precisions)) if precisions else 0.0,
            "mean_recall_at_5": float(np.mean(recalls)) if recalls else 0.0,
            "mean_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "unanswerable_stats": {
                "total_unanswerable": unans_count,
                "returned_evidence": returned_evidence_count,
                "returned_insufficient_evidence": insufficient_evidence_count
            }
        }

    # 1. Overall
    overall_summary = aggregate_group(records)

    # 2. By Retrieval Mode
    by_mode = {}
    for mode in ["vector", "graph", "hybrid"]:
        subset = [r for r in records if r["retrieval_mode"] == mode]
        by_mode[mode] = aggregate_group(subset)

    # 3. By Category
    by_category = {}
    categories = set(r["category"] for r in records)
    for cat in categories:
        subset = [r for r in records if r["category"] == cat]
        by_category[cat] = aggregate_group(subset)

    return {
        "overall": overall_summary,
        "by_retrieval_mode": by_mode,
        "by_category": by_category
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
