"""
Corrected Retrieval Evaluation Runner (v2) for clinical GraphRAG repository.
Executes benchmark questions against Vector, Graph, and Hybrid retrieval modes,
calculates updated metrics, records path recovery, aggregates by subset, and saves results.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag.evaluation.runner_v2")

def calculate_metrics_v2(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> Dict[str, float]:
    if not expected_ids:
        # Unanswerable
        return {
            "precision_at_5": 1.0 if not retrieved_ids else 0.0,
            "recall_at_5": 1.0 if not retrieved_ids else 0.0,
            "mrr": 1.0 if not retrieved_ids else 0.0
        }
    
    retrieved_k = retrieved_ids[:k]
    expected_set = set(expected_ids)
    hits = sum(1 for rid in retrieved_k if rid in expected_set)
    
    # Precision@5: relevant retrieved / 5, capped by actual number retrieved if fewer than 5
    denom = len(retrieved_k) if len(retrieved_k) < k else k
    precision = (hits / denom) if denom > 0 else 0.0
    
    # Recall@5
    recall = (hits / len(expected_set)) if len(expected_set) > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in expected_set:
            mrr = 1.0 / (idx + 1)
            break
            
    return {
        "precision_at_5": precision,
        "recall_at_5": recall,
        "mrr": mrr
    }

def verify_strict_path_recovery(q_id: str, graph_paths: List[Dict[str, Any]]) -> bool:
    """
    Strictly matches query graph paths against benchmark ground truth definitions.
    Ensures correct nodes, edge types, hop count, and contiguous connectivity.
    """
    if q_id == "q5":
        for p in graph_paths:
            if " -> " not in p.get("relationship", ""):
                src = p.get("source")
                tgt = p.get("target")
                rel = p.get("relationship")
                if (src == "Mezereum" and tgt == "Pruritus" and rel == "TREATS") or \
                   (src == "Pruritus" and tgt == "Mezereum" and rel == "TREATS"):
                    return True
        return False
    elif q_id == "q6":
        for p in graph_paths:
            src = p.get("source")
            tgt = p.get("target")
            rel = p.get("relationship", "")
            if (src == "Cyclosporine" and tgt == "Rheumatoid Arthritis" and rel == "INTERACTS_WITH -> Naproxen -> TREATS") or \
               (src == "Rheumatoid Arthritis" and tgt == "Cyclosporine" and rel == "TREATS -> Naproxen -> INTERACTS_WITH"):
                return True
        return False
    elif q_id == "q7":
        for p in graph_paths:
            src = p.get("source")
            tgt = p.get("target")
            rel = p.get("relationship", "")
            if (src == "Aspirin" and tgt == "Gout" and rel == "INTERACTS_WITH -> Naproxen -> TREATS") or \
               (src == "Gout" and tgt == "Aspirin" and rel == "TREATS -> Naproxen -> INTERACTS_WITH"):
                return True
        return False
    elif q_id == "q8":
        has_quin_cyclo_nap = False
        has_cyclo_nap_gout = False
        for p in graph_paths:
            src = p.get("source")
            tgt = p.get("target")
            rel = p.get("relationship", "")
            if (src == "Quinolones" and tgt == "Naproxen" and rel == "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH") or \
               (src == "Naproxen" and tgt == "Quinolones" and rel == "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH"):
                has_quin_cyclo_nap = True
            if (src == "Cyclosporine" and tgt == "Gout" and rel == "INTERACTS_WITH -> Naproxen -> TREATS") or \
               (src == "Gout" and tgt == "Cyclosporine" and rel == "TREATS -> Naproxen -> INTERACTS_WITH"):
                has_cyclo_nap_gout = True
        return has_quin_cyclo_nap and has_cyclo_nap_gout
    elif q_id == "q9":
        has_quin_cyclo_nap = False
        has_cyclo_nap_asp = False
        for p in graph_paths:
            src = p.get("source")
            tgt = p.get("target")
            rel = p.get("relationship", "")
            if (src == "Quinolones" and tgt == "Naproxen" and rel == "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH") or \
               (src == "Naproxen" and tgt == "Quinolones" and rel == "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH"):
                has_quin_cyclo_nap = True
            if (src == "Cyclosporine" and tgt == "Aspirin" and rel == "INTERACTS_WITH -> Naproxen -> INTERACTS_WITH") or \
               (src == "Aspirin" and tgt == "Cyclosporine" and rel == "INTERACTS_WITH -> Naproxen -> INTERACTS_WITH"):
                has_cyclo_nap_asp = True
        return has_quin_cyclo_nap and has_cyclo_nap_asp
    return False

def run_evaluation_v2():
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
        expected_ents = item.get("expected_entities", [])
        expected_rel_types = item.get("expected_relationship_types", [])
        expected_hop_count = item.get("expected_hop_count", 0)

        # Detect query entities for logging
        detected_ents = [e["normalized_name"] for e in extract_query_entities(question)]

        # --- 1. Vector RAG ---
        t0 = time.perf_counter()
        try:
            vector_items = vector_retrieve(question, top_k=5)
            vector_retrieved_ids = []
            for vi in vector_items:
                vector_retrieved_ids.extend(vi.source_ids)
            vector_retrieved_ids = list(dict.fromkeys(vector_retrieved_ids))
            vector_status = "success"
        except Exception as e:
            logger.error(f"Vector failed for {q_id}: {str(e)}")
            vector_retrieved_ids = []
            vector_status = f"failed: {str(e)}"
        t1 = time.perf_counter()
        vector_latency = (t1 - t0) * 1000.0
        vector_metrics = calculate_metrics_v2(vector_retrieved_ids, expected_ids)

        records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "vector",
            "detected_entities": detected_ents,
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": vector_retrieved_ids,
            "number_of_retrieved_evidence_items": len(vector_retrieved_ids),
            "graph_paths": [],
            "expected_graph_path": "",
            "path_recovered": False,
            "precision_at_5": vector_metrics["precision_at_5"],
            "recall_at_5": vector_metrics["recall_at_5"],
            "mrr": vector_metrics["mrr"],
            "retrieval_latency_ms": vector_latency,
            "total_latency_ms": vector_latency,
            "answer_text": "",
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
            graph_status = graph_res.get("status", "ok")
        except Exception as e:
            logger.error(f"Graph failed for {q_id}: {str(e)}")
            graph_retrieved_ids = []
            graph_paths = []
            graph_status = f"failed: {str(e)}"
        t1 = time.perf_counter()
        graph_latency = (t1 - t0) * 1000.0
        graph_metrics = calculate_metrics_v2(graph_retrieved_ids, expected_ids)

        # Path recovery detection for q5-q9
        expected_path_desc = ""
        if q_id == "q5":
            expected_path_desc = "Mezereum -[TREATS]-> Pruritus"
        elif q_id == "q6":
            expected_path_desc = "Rheumatoid Arthritis <-[TREATS]- Naproxen -[INTERACTS_WITH]-> Cyclosporine"
        elif q_id == "q7":
            expected_path_desc = "Gout <-[TREATS]- Naproxen -[INTERACTS_WITH]-> Aspirin"
        elif q_id == "q8":
            expected_path_desc = "Gout <-[TREATS]- Naproxen -[INTERACTS_WITH]-> Cyclosporine <-[INTERACTS_WITH]- Quinolones"
        elif q_id == "q9":
            expected_path_desc = "Aspirin <-[INTERACTS_WITH]- Naproxen -[INTERACTS_WITH]-> Cyclosporine <-[INTERACTS_WITH]- Quinolones"

        path_recovered = verify_strict_path_recovery(q_id, graph_paths)

        records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "graph",
            "detected_entities": detected_ents,
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": graph_retrieved_ids,
            "number_of_retrieved_evidence_items": len(graph_retrieved_ids),
            "graph_paths": graph_paths,
            "expected_graph_path": expected_path_desc,
            "path_recovered": path_recovered,
            "precision_at_5": graph_metrics["precision_at_5"],
            "recall_at_5": graph_metrics["recall_at_5"],
            "mrr": graph_metrics["mrr"],
            "retrieval_latency_ms": graph_latency,
            "total_latency_ms": graph_latency,
            "answer_text": "",
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
            hybrid_status = hybrid_res.get("status", "ok")
        except Exception as e:
            logger.error(f"Hybrid failed for {q_id}: {str(e)}")
            hybrid_retrieved_ids = []
            hybrid_paths = []
            hybrid_status = f"failed: {str(e)}"
        t1 = time.perf_counter()
        hybrid_latency = (t1 - t0) * 1000.0
        hybrid_metrics = calculate_metrics_v2(hybrid_retrieved_ids, expected_ids)

        records.append({
            "question_id": q_id,
            "category": category,
            "answerable": answerable,
            "retrieval_mode": "hybrid",
            "detected_entities": detected_ents,
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": hybrid_retrieved_ids,
            "number_of_retrieved_evidence_items": len(hybrid_retrieved_ids),
            "graph_paths": hybrid_paths,
            "expected_graph_path": expected_path_desc,
            "path_recovered": path_recovered,
            "precision_at_5": hybrid_metrics["precision_at_5"],
            "recall_at_5": hybrid_metrics["recall_at_5"],
            "mrr": hybrid_metrics["mrr"],
            "retrieval_latency_ms": hybrid_latency,
            "total_latency_ms": hybrid_latency,
            "answer_text": "",
            "status": hybrid_status,
            "unanswerable_eval": {
                "evidence_retrieved": len(hybrid_retrieved_ids) > 0,
                "insufficient_evidence_state": (hybrid_status == "insufficient_evidence")
            }
        })

    # Save retrieval_results_v2.json
    results_path = os.path.join(results_dir, "retrieval_results_v2.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Compute Summary
    summary = compute_summary_v2(records)
    summary_path = os.path.join(results_dir, "retrieval_summary_v2.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("V2 Evaluation completed successfully.")

def compute_summary_v2(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Separately identify warm runs (exclude very first execution which has cold start initialization)
    # The first run in temporal order is index 0: Vector q1
    warm_records = records[1:]
    
    def agg_subset(subset: List[Dict[str, Any]], warm_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        precisions = [r["precision_at_5"] for r in subset]
        recalls = [r["recall_at_5"] for r in subset]
        mrrs = [r["mrr"] for r in subset]
        
        # Latencies including cold start
        latencies = [r["retrieval_latency_ms"] for r in subset]
        
        # Latencies warm run
        warm_latencies = [r["retrieval_latency_ms"] for r in warm_subset]

        unans_subset = [r for r in subset if not r["answerable"]]
        unans_count = len(unans_subset)
        returned_evidence = sum(1 for r in unans_subset if r["unanswerable_eval"]["evidence_retrieved"])
        insufficient_evidence = sum(1 for r in unans_subset if r["unanswerable_eval"]["insufficient_evidence_state"])

        return {
            "mean_precision_at_5": float(np.mean(precisions)) if precisions else 0.0,
            "mean_recall_at_5": float(np.mean(recalls)) if recalls else 0.0,
            "mean_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "overall_median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
            "overall_p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "warm_median_latency_ms": float(np.median(warm_latencies)) if warm_latencies else 0.0,
            "warm_p95_latency_ms": float(np.percentile(warm_latencies, 95)) if warm_latencies else 0.0,
            "unanswerable_stats": {
                "total_unanswerable": unans_count,
                "returned_evidence": returned_evidence,
                "returned_insufficient_evidence": insufficient_evidence
            }
        }

    # 1. Overall
    overall = agg_subset(records, warm_records)

    # 2. By Retrieval Mode
    by_mode = {}
    for mode in ["vector", "graph", "hybrid"]:
        sub = [r for r in records if r["retrieval_mode"] == mode]
        w_sub = [r for r in warm_records if r["retrieval_mode"] == mode]
        by_mode[mode] = agg_subset(sub, w_sub)

    # 3. By Category
    by_category = {}
    cats = sorted(list(set(r["category"] for r in records)))
    for cat in cats:
        sub = [r for r in records if r["category"] == cat]
        w_sub = [r for r in warm_records if r["category"] == cat]
        by_category[cat] = agg_subset(sub, w_sub)

    # 4. Custom Subsets
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

    # 5. Answerable-only
    answerable_only = {}
    for mode in ["vector", "graph", "hybrid"]:
        sub = [r for r in records if r["answerable"] and r["retrieval_mode"] == mode]
        w_sub = [r for r in warm_records if r["answerable"] and r["retrieval_mode"] == mode]
        answerable_only[mode] = agg_subset(sub, w_sub)

    return {
        "overall": overall,
        "by_retrieval_mode": by_mode,
        "by_category": by_category,
        "subsets": custom_subsets,
        "overall_answerable_only": answerable_only
    }

if __name__ == "__main__":
    run_evaluation_v2()
