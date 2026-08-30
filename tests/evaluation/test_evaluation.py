"""
Unit tests for the benchmark dataset and evaluation runner foundation.
"""
import os
import json
import pytest
from evaluation.run_evaluation import calculate_metrics, run_evaluation

BENCHMARK_PATH = "evaluation/benchmark/benchmark_set.json"

def test_benchmark_schema_validation():
    """
    Validates that the benchmark_set.json file exists and complies with the expected schema.
    """
    assert os.path.exists(BENCHMARK_PATH), f"Benchmark set file not found at {BENCHMARK_PATH}"
    
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "benchmark_items" in data
    assert isinstance(data["benchmark_items"], list)
    assert len(data["benchmark_items"]) > 0
    
    allowed_categories = {
        "direct", "semantic", "relationship", "two_hop",
        "multi_hop", "interaction", "contraindication", "citation", "unanswerable"
    }
    
    for item in data["benchmark_items"]:
        assert "id" in item
        assert "question" in item
        assert "category" in item
        assert item["category"] in allowed_categories
        assert "expected_source_ids" in item
        assert isinstance(item["expected_source_ids"], list)
        assert "expected_entities" in item
        assert isinstance(item["expected_entities"], list)
        assert "expected_relationship_ids" in item
        assert isinstance(item["expected_relationship_ids"], list)
        assert "expected_relationship_types" in item
        assert isinstance(item["expected_relationship_types"], list)
        assert "expected_hop_count" in item
        assert isinstance(item["expected_hop_count"], int)
        assert "answerable" in item
        assert isinstance(item["answerable"], bool)

def test_answerable_unanswerable_items():
    """
    Validates the count and distribution of answerable and unanswerable questions.
    """
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    items = data["benchmark_items"]
    answerable_items = [i for i in items if i["answerable"]]
    unanswerable_items = [i for i in items if not i["answerable"]]
    
    assert len(items) == 20
    assert len(answerable_items) == 15
    assert len(unanswerable_items) == 5
    
    for item in unanswerable_items:
        assert item["category"] == "unanswerable"
        assert len(item["expected_source_ids"]) == 0
        assert len(item["expected_entities"]) == 0

def test_source_id_validation():
    """
    Validates that expected source IDs correspond to valid document IDs in documents.json.
    """
    with open("data/processed/documents.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    valid_doc_ids = {d["metadata"]["document_id"] for d in docs}
    
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data["benchmark_items"]:
        if item["answerable"]:
            for sid in item["expected_source_ids"]:
                assert sid in valid_doc_ids, f"Source ID {sid} not found in documents.json"

def test_metrics_calculation():
    """
    Tests calculation of Precision@K, Recall@K, and MRR.
    """
    # 1. Answerable case
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    expected = ["doc2", "doc6"]
    
    metrics = calculate_metrics(retrieved, expected, k=5)
    assert metrics["precision_at_5"] == 0.2  # 1 hit (doc2) / 5
    assert metrics["recall_at_5"] == 0.5     # 1 hit / 2 expected
    assert metrics["mrr"] == 0.5             # first hit at index 1 -> rank 2 -> reciprocal 1/2
    
    # 2. Unanswerable case (success when nothing retrieved)
    metrics_unans = calculate_metrics([], [], k=5)
    assert metrics_unans["precision_at_5"] == 1.0
    assert metrics_unans["recall_at_5"] == 1.0
    assert metrics_unans["mrr"] == 1.0

def test_evaluation_runner_serialization(tmp_path):
    """
    Verifies the evaluation runner can execute a dry run/mock run, compute metrics, and serialize results.
    """
    # Create a small mock benchmark set
    mock_benchmark = {
        "benchmark_items": [
            {
                "id": "mock_q1",
                "question": "What are the indications listed for Ofloxacin ophthalmic solution?",
                "category": "direct",
                "expected_source_ids": ["f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff"],
                "expected_entities": ["drug_ofloxacin_ophthalmic"],
                "expected_relationship_ids": [],
                "expected_relationship_types": [],
                "expected_hop_count": 0,
                "answerable": True
            }
        ]
    }
    
    mock_benchmark_path = os.path.join(tmp_path, "mock_benchmark.json")
    with open(mock_benchmark_path, "w", encoding="utf-8") as f:
        json.dump(mock_benchmark, f)
        
    results_dir = os.path.join(tmp_path, "results")
    
    # Run evaluation on mock benchmark
    report = run_evaluation(benchmark_path=mock_benchmark_path, results_dir=results_dir)
    
    assert "results" in report
    assert "summary" in report
    assert len(report["results"]) == 3  # vector, graph, hybrid
    
    # Check that output files are saved to the specified results directory
    files = os.listdir(results_dir)
    assert len(files) == 2
    assert "retrieval_results.json" in files
    assert "retrieval_summary.json" in files
