"""
Unit tests for strict graph path recovery logic in verify_strict_path_recovery.
"""
from evaluation.run_evaluation_v2 import verify_strict_path_recovery

def test_q5_recovery():
    # Exact forward path
    paths = [{"source": "Mezereum", "relationship": "TREATS", "target": "Pruritus"}]
    assert verify_strict_path_recovery("q5", paths) is True

    # Exact reverse traversal
    paths_rev = [{"source": "Pruritus", "relationship": "TREATS", "target": "Mezereum"}]
    assert verify_strict_path_recovery("q5", paths_rev) is True

    # Incorrect relationship type
    paths_wrong_rel = [{"source": "Mezereum", "relationship": "INTERACTS_WITH", "target": "Pruritus"}]
    assert verify_strict_path_recovery("q5", paths_wrong_rel) is False

    # Incorrect destination
    paths_wrong_dest = [{"source": "Mezereum", "relationship": "TREATS", "target": "Diabetes"}]
    assert verify_strict_path_recovery("q5", paths_wrong_dest) is False

def test_q6_q7_recovery():
    # q6 exact forward
    q6_forward = [{"source": "Cyclosporine", "target": "Rheumatoid Arthritis", "relationship": "INTERACTS_WITH -> Naproxen -> TREATS"}]
    assert verify_strict_path_recovery("q6", q6_forward) is True

    # q6 exact reverse
    q6_reverse = [{"source": "Rheumatoid Arthritis", "target": "Cyclosporine", "relationship": "TREATS -> Naproxen -> INTERACTS_WITH"}]
    assert verify_strict_path_recovery("q6", q6_reverse) is True

    # q6 incorrect relationship type
    q6_wrong = [{"source": "Cyclosporine", "target": "Rheumatoid Arthritis", "relationship": "INTERACTS_WITH -> Naproxen -> CONTRAINDICATED_FOR"}]
    assert verify_strict_path_recovery("q6", q6_wrong) is False

    # q7 exact forward
    q7_forward = [{"source": "Aspirin", "target": "Gout", "relationship": "INTERACTS_WITH -> Naproxen -> TREATS"}]
    assert verify_strict_path_recovery("q7", q7_forward) is True

    # q7 exact reverse
    q7_reverse = [{"source": "Gout", "target": "Aspirin", "relationship": "TREATS -> Naproxen -> INTERACTS_WITH"}]
    assert verify_strict_path_recovery("q7", q7_reverse) is True

def test_q8_q9_recovery():
    # q8 complete contiguous paths
    q8_valid = [
        {"source": "Quinolones", "target": "Naproxen", "relationship": "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH"},
        {"source": "Cyclosporine", "target": "Gout", "relationship": "INTERACTS_WITH -> Naproxen -> TREATS"}
    ]
    assert verify_strict_path_recovery("q8", q8_valid) is True

    # q8 disconnected/missing part
    q8_invalid = [
        {"source": "Quinolones", "target": "Naproxen", "relationship": "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH"}
    ]
    assert verify_strict_path_recovery("q8", q8_invalid) is False

    # q9 complete contiguous paths
    q9_valid = [
        {"source": "Quinolones", "target": "Naproxen", "relationship": "INTERACTS_WITH -> Cyclosporine -> INTERACTS_WITH"},
        {"source": "Cyclosporine", "target": "Aspirin", "relationship": "INTERACTS_WITH -> Naproxen -> INTERACTS_WITH"}
    ]
    assert verify_strict_path_recovery("q9", q9_valid) is True
