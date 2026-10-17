import pytest

from bench.analyze import (
    analyze_decisions,
    analyze_guard,
    auc,
    expected_calibration_error,
    majority_prediction,
    mcnemar_exact,
)


def test_exact_mcnemar_uses_only_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 0) == 0.25
    assert mcnemar_exact(2, 2) == 1.0


def test_auc_handles_ties_and_missing_classes():
    assert auc([0.5, 0.9], [0.1, 0.5]) == 0.875
    assert auc([0.5], [0.5]) == 0.5
    assert auc([], [0.1]) is None
    assert auc([0.9], []) is None


def test_ece_uses_confidence_and_correctness_bins():
    rows = [
        {"conf": 0.9, "correct": True},
        {"conf": 0.8, "correct": False},
        {"conf": 0.2, "correct": False},
    ]
    assert expected_calibration_error(rows, bins=2) == pytest.approx(0.3)


def test_majority_prediction_counts_votes_and_resolves_ties_by_first_vote():
    assert majority_prediction([{"pred": True}, {"pred": False}, {"pred": True}]) is True
    assert majority_prediction([{"pred": False}, {"pred": True}]) is False


def test_decision_analysis_reports_majority_mcnemar_and_paired_ci():
    rows = [
        {"id": "a", "kind": "noul", "model": "old", "expected": True, "pred": True,
         "correct": True, "conf": 0.8, "p_true": 0.8, "p_correct": 0.8},
        {"id": "a", "kind": "noul", "model": "old", "expected": True, "pred": False,
         "correct": False, "conf": 0.9, "p_true": 0.1, "p_correct": 0.1},
        {"id": "b", "kind": "noul", "model": "old", "expected": False, "pred": True,
         "correct": False, "conf": 0.7, "p_true": 0.7, "p_correct": 0.3},
        {"id": "a", "kind": "noul", "model": "new", "expected": True, "pred": True,
         "correct": True, "conf": 0.9, "p_true": 0.9, "p_correct": 0.9},
        {"id": "a", "kind": "noul", "model": "new", "expected": True, "pred": True,
         "correct": True, "conf": 0.9, "p_true": 0.9, "p_correct": 0.9},
        {"id": "b", "kind": "noul", "model": "new", "expected": False, "pred": False,
         "correct": True, "conf": 0.8, "p_true": 0.2, "p_correct": 0.8},
    ]

    result = analyze_decisions(rows, bootstrap_samples=200, seed=4)

    assert result["models"]["old"]["accuracy_all_reps"] == 1 / 3
    assert result["models"]["old"]["accuracy_majority"] == 1 / 2
    assert result["models"]["old"]["brier_noul"] == pytest.approx(1.34 / 3)
    assert result["pair"]["mcnemar"]["only_first_correct"] == 0
    assert result["pair"]["mcnemar"]["only_second_correct"] == 1
    assert result["pair"]["flipped_cases"] == ["b"]
    assert result["pair"]["log_loss_diff_ci"][0] <= result["pair"]["log_loss_diff"]
    assert result["pair"]["log_loss_diff"] <= result["pair"]["log_loss_diff_ci"][1]


def test_guard_analysis_averages_reps_and_reports_wording_metrics():
    rows = [
        {"id": "positive", "model": "m", "category": "attack", "injection": True,
         "self_claim": False, "text": "attack", "p_current": 0.7, "p_self_claim": 0.1},
        {"id": "benign", "model": "m", "category": "ordinary", "injection": False,
         "self_claim": False, "text": "hello", "p_current": 0.3, "p_self_claim": 0.2},
        {"id": "benign", "model": "m", "category": "ordinary", "injection": False,
         "self_claim": False, "text": "hello", "p_current": 0.5, "p_self_claim": 0.2},
        {"id": "claim", "model": "m", "category": "claim", "injection": True,
         "self_claim": True, "text": "safe", "p_current": 0.4, "p_self_claim": 0.8},
    ]

    result = analyze_guard(rows)

    current = result["groups"][("m", "current")]
    assert current["recall"] == 0.5
    assert current["false_positive_rate"] == 0.0
    assert current["benign_ge_0_2"] == 1
    assert current["auc"] == 0.75
    claim = result["groups"][("m", "self_claim")]
    assert claim["recall"] == 1.0
    assert claim["false_positive_rate"] == 0.0
    assert result["errors"]

