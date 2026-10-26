"""Synthetic checks for locked Phase 15 Holdout metrics and resampling."""

import numpy as np

from scripts.evaluate_phase15_holdout import average_precision, bootstrap, measures


def test_perfect_multilabel_scores_and_tied_ap():
    labels = np.vstack([np.zeros((3, 10), dtype=np.int8), np.ones((3, 10), dtype=np.int8)])
    probabilities = np.where(labels == 1, 0.9, 0.1)
    report = measures(probabilities, labels, ["p1", "p1", "p2", "p3", "p4", "p4"])
    assert report["macro_auc"] == 1.0
    assert report["macro_ap"] == 1.0
    assert report["macro_f1_at_0_5"] == 1.0
    assert report["per_class"][0]["positive_patients"] == 2
    assert average_precision(np.array([1, 0, 1, 0]), np.array([0.8, 0.8, 0.2, 0.2])) == 0.5


def test_patient_bootstrap_retries_invalid_draws_and_preserves_pairing():
    labels = np.vstack([np.zeros((2, 10), dtype=np.int8), np.ones((2, 10), dtype=np.int8)])
    a = np.where(labels == 1, 0.8, 0.2)
    scores = {"A": a, "B": a.copy(), "fusion": a.copy()}
    report = bootstrap(scores, labels, ["negative", "negative", "positive", "positive"],
                       replicates=20, seed=42)
    assert report["valid_replicates"] == 20
    assert report["invalid_resamples"] > 0
    assert report["paired_difference_ci95"]["fusion_minus_A"] == [0.0, 0.0]
