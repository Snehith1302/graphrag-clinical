"""
Final Retrieval Evaluation Runner for clinical GraphRAG repository.
Executes benchmark questions against Vector, Graph, and Hybrid retrieval modes,
calculates final metrics, path recovery, aggregates by subset, and saves results.
"""
import os
import json
import time
import logging
import numpy as np
from typing import Dict, Any, List

from backend.app.retrieval.vector_retrieve import vector_retrieve
from backend.app.retrieval.graph_retrieve import graph_retrieve, extract_query_entities
from backend.app.retrieval.hybrid_retrieve import hybrid_retrieve
from run_evaluation_v2 import verify_strict_path_recovery, calculate_metrics_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag.evaluation.runner_final")

def run_evaluation_final():
    benchmark_path = "evaluation/benchmark/benchmark_set.json"
    results_dir = "evaluation/results"
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    items = benchmark_data["benchmark_items"]
    records = []
    success_count = 0
    fail_count = 0

    os.makedirs(results_dir, exist_ok=True)

    for item in items:
        q_id = item["id"]
        question = item["question"]
        category = item["category"]
        expected_ids = item["expected_source_ids"]
        answerable = item["answerable"]

        # --- 1. Vector RAG ---
        t0 = time.perf_counter()
        try:
            vector_items = vector_retrieve(question, top_k=5)
            vector_retrieved_ids = []
            for vi in vector_items:
                vector_retrieved_ids.extend(vi.source_ids)
            vector_retrieved_ids = list(dict.fromkeys(vector_retrieved_ids))
            
            # Serialize evidence items to string representations for saving
            vector_ev = [{"content": item.content, "source_ids": item.source_ids} for item in vector_items]
            vector_status = "success"
            success_count += 1
        except Exception as e:
            logger.error(f"Vector failed for {q_id}: {str(e)}")
            vector_retrieved_ids = []
            vector_ev = []
            vector_status = f"failed: {str(e)}"
            fail_count += 1
        t1 = time.perf_counter()
        vector_latency = (t1 - t0) * 1000.0
        vector_metrics = calculate_metrics_v2(vector_retrieved_ids, expected_ids)

        records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "vector",
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": vector_retrieved_ids,
            "evidence_items": vector_ev,
            "graph_paths": [],
            "Precision@5": vector_metrics["precision_at_5"],
            "Recall@5": vector_metrics["recall_at_5"],
            "MRR": vector_metrics["mrr"],
            "retrieval_latency_ms": vector_latency,
            "total_latency_ms": vector_latency,
            "status": vector_status,
            "unanswerable_eval": {
                "evidence_retrieved": len(vector_retrieved_ids) > 0,
                "insufficient_evidence_state": len(vector_retrieved_ids) == 0
            }
        })

        # --- 2. Graph RAG ---
        t0 = time.perf_counter()
        try:
            graph_res = graph_retrieve(question)
            graph_retrieved_ids = graph_res.get("source_ids", [])
            graph_paths = graph_res.get("graph_paths", [])
            graph_items = graph_res.get("evidence_items", [])
            graph_ev = [{"content": item.content, "source_ids": item.source_ids} for item in graph_items]
            graph_status = graph_res.get("status", "ok")
            success_count += 1
        except Exception as e:
            logger.error(f"Graph failed for {q_id}: {str(e)}")
            graph_retrieved_ids = []
            graph_paths = []
            graph_ev = []
            graph_status = f"failed: {str(e)}"
            fail_count += 1
        t1 = time.perf_counter()
        graph_latency = (t1 - t0) * 1000.0
        graph_metrics = calculate_metrics_v2(graph_retrieved_ids, expected_ids)
        path_recovered = verify_strict_path_recovery(q_id, graph_paths)

        records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "graph",
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": graph_retrieved_ids,
            "evidence_items": graph_ev,
            "graph_paths": graph_paths,
            "path_recovered": path_recovered,
            "Precision@5": graph_metrics["precision_at_5"],
            "Recall@5": graph_metrics["recall_at_5"],
            "MRR": graph_metrics["mrr"],
            "retrieval_latency_ms": graph_latency,
            "total_latency_ms": graph_latency,
            "status": graph_status,
            "unanswerable_eval": {
                "evidence_retrieved": len(graph_retrieved_ids) > 0,
                "insufficient_evidence_state": (graph_status == "insufficient_evidence")
            }
        })

        # --- 3. Hybrid RAG ---
        t0 = time.perf_counter()
        try:
            hybrid_res = hybrid_retrieve(question, top_k=5)
            hybrid_retrieved_ids = hybrid_res.get("source_ids", [])
            hybrid_paths = hybrid_res.get("graph_paths", [])
            hybrid_items = hybrid_res.get("evidence_items", [])
            hybrid_ev = [{"content": item.content, "source_ids": item.source_ids} for item in hybrid_items]
            hybrid_status = hybrid_res.get("status", "ok")
            success_count += 1
        except Exception as e:
            logger.error(f"Hybrid failed for {q_id}: {str(e)}")
            hybrid_retrieved_ids = []
            hybrid_paths = []
            hybrid_ev = []
            hybrid_status = f"failed: {str(e)}"
            fail_count += 1
        t1 = time.perf_counter()
        hybrid_latency = (t1 - t0) * 1000.0
        hybrid_metrics = calculate_metrics_v2(hybrid_retrieved_ids, expected_ids)

        records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "hybrid",
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": hybrid_retrieved_ids,
            "evidence_items": hybrid_ev,
            "graph_paths": hybrid_paths,
            "path_recovered": path_recovered,
            "Precision@5": hybrid_metrics["precision_at_5"],
            "Recall@5": hybrid_metrics["recall_at_5"],
            "MRR": hybrid_metrics["mrr"],
            "retrieval_latency_ms": hybrid_latency,
            "total_latency_ms": hybrid_latency,
            "status": hybrid_status,
            "unanswerable_eval": {
                "evidence_retrieved": len(hybrid_retrieved_ids) > 0,
                "insufficient_evidence_state": (hybrid_status == "insufficient_evidence")
            }
        })

    # Save retrieval_results_final.json
    results_path = os.path.join(results_dir, "retrieval_results_final.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Compute Summary
    summary = compute_summary_final(records, success_count, fail_count)
    summary_path = os.path.join(results_dir, "retrieval_summary_final.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Final evaluation completed successfully.")

def compute_summary_final(records: List[Dict[str, Any]], success_count: int, fail_count: int) -> Dict[str, Any]:
    warm_records = records[1:]
    
    def agg_subset(subset: List[Dict[str, Any]], warm_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        precisions = [r["Precision@5"] for r in subset]
        recalls = [r["Recall@5"] for r in subset]
        mrrs = [r["MRR"] for r in subset]
        latencies = [r["retrieval_latency_ms"] for r in subset]
        warm_latencies = [r["retrieval_latency_ms"] for r in warm_subset]

        unans_subset = [r for r in subset if not r["answerable"]]
        unans_count = len(unans_subset)
        returned_evidence = sum(1 for r in unans_subset if r["unanswerable_eval"]["evidence_retrieved"])
        insufficient_evidence = sum(1 for r in unans_subset if r["unanswerable_eval"]["insufficient_evidence_state"])

        # Path recovery rate for graph/hybrid
        path_recovered_count = sum(1 for r in subset if r.get("path_recovered", False))

        return {
            "mean_precision_at_5": float(np.mean(precisions)) if precisions else 0.0,
            "mean_recall_at_5": float(np.mean(recalls)) if recalls else 0.0,
            "mean_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "overall_median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
            "overall_p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "warm_median_latency_ms": float(np.median(warm_latencies)) if warm_latencies else 0.0,
            "warm_p95_latency_ms": float(np.percentile(warm_latencies, 95)) if warm_latencies else 0.0,
            "path_recovery_count": path_recovered_count,
            "unanswerable_stats": {
                "total_unanswerable": unans_count,
                "returned_evidence": returned_evidence,
                "returned_insufficient_evidence": insufficient_evidence
            }
        }

    # 1. Overall by Retrieval Mode
    by_mode = {}
    for mode in ["vector", "graph", "hybrid"]:
        sub = [r for r in records if r["retrieval_mode"] == mode]
        w_sub = [r for r in warm_records if r["retrieval_mode"] == mode]
        by_mode[mode] = agg_subset(sub, w_sub)

    # 2. Answerable-only
    answerable_only = {}
    for mode in ["vector", "graph", "hybrid"]:
        sub = [r for r in records if r["answerable"] and r["retrieval_mode"] == mode]
        w_sub = [r for r in warm_records if r["answerable"] and r["retrieval_mode"] == mode]
        answerable_only[mode] = agg_subset(sub, w_sub)

    # 3. Custom Subsets
    custom_subsets = {}
    subsets_def = {
        "direct_plus_semantic": ["direct", "semantic"],
        "relationship_heavy": ["relationship", "interaction", "contraindication"],
        "multi_hop_subset": ["two_hop", "multi_hop"],
        "citation": ["citation"],
        "unanswerable": ["unanswerable"]
    }
    for name, cats_list in subsets_def.items():
        custom_subsets[name] = {}
        for mode in ["vector", "graph", "hybrid"]:
            sub = [r for r in records if r["category"] in cats_list and r["retrieval_mode"] == mode]
            w_sub = [r for r in warm_records if r["category"] in cats_list and r["retrieval_mode"] == mode]
            custom_subsets[name][mode] = agg_subset(sub, w_sub)

    # Path recovery for q5-q9 specifically
    q5_q9_list = ["q5", "q6", "q7", "q8", "q9"]
    q5_q9_path_recovery = {}
    for mode in ["graph", "hybrid"]:
        sub = [r for r in records if r["question_id"] in q5_q9_list and r["retrieval_mode"] == mode]
        q5_q9_path_recovery[mode] = sum(1 for r in sub if r.get("path_recovered", False))

    # Evidence retrieval success rate & unanswerable rejection rate
    unans_subset = [r for r in records if not r["answerable"]]
    total_unans = len(unans_subset)
    total_unans_rejected = sum(1 for r in unans_subset if r["unanswerable_eval"]["insufficient_evidence_state"])
    unans_rejection_rate = (total_unans_rejected / total_unans) if total_unans > 0 else 0.0

    ans_subset = [r for r in records if r["answerable"]]
    total_ans = len(ans_subset)
    total_ans_recalled = sum(1 for r in ans_subset if r["Recall@5"] > 0)
    evidence_retrieval_success_rate = (total_ans_recalled / total_ans) if total_ans > 0 else 0.0

    return {
        "success_count": success_count,
        "failed_count": fail_count,
        "evidence_retrieval_success_rate": evidence_retrieval_success_rate,
        "unanswerable_rejection_rate": unans_rejection_rate,
        "by_retrieval_mode": by_mode,
        "overall_answerable_only": answerable_only,
        "subsets": custom_subsets,
        "q5_q9_path_recovery": q5_q9_path_recovery
    }

if __name__ == "__main__":
    run_evaluation_final()
