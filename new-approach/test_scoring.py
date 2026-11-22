from math import sqrt

import numpy as np
import pytest

from scoring import paired_site_difference, select_threshold, site_metrics


def test_threshold_maximizes_pooled_f1():
    probabilities = np.array([[0.95, 0.8, 0.7], [0.3, 0.2, 0.1]], dtype=np.float32)
    targets = np.array([[0, 1, 1], [0, 0, 0]])
    threshold = select_threshold(probabilities, targets)
    assert threshold == float(probabilities[0, 2])
    assert site_metrics(probabilities >= threshold, targets)["f1"] == pytest.approx(0.8)


def test_threshold_ties_and_all_negative_validation():
    probabilities = np.array([[0.9, 0.5, 0.5, 0.5]])
    targets = np.array([[1, 1, 0, 0]])
    assert select_threshold(probabilities, targets) == 0.9
    assert select_threshold(probabilities[:, ::-1], targets[:, ::-1]) == 0.9

    probabilities = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    threshold = select_threshold(probabilities, np.zeros_like(probabilities))
    assert not np.any(probabilities >= threshold)


def test_site_metrics_count_sites_and_exact_sets():
    first = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]])
    targets = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 1]])
    assert site_metrics(first, targets) == {
        "fp": 1,
        "fn": 1,
        "tp": 3,
        "tn": 4,
        "shots": 3,
        "precision": 0.75,
        "recall": 0.75,
        "f1": 0.75,
        "site_error_rate": 2 / 9,
        "exact_set_accuracy": 2 / 3,
    }

    empty = np.zeros((2, 2, 2), dtype=bool)
    metrics = site_metrics(empty, empty)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert metrics["exact_set_accuracy"] == 1


def test_paired_difference_clusters_sites_by_shot():
    first = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]])
    second = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    targets = np.zeros_like(first)
    result = paired_site_difference(first, second, targets)
    margin = 1.959963984540054 * sqrt(7) / 12
    assert result["difference"] == pytest.approx(1 / 12)
    assert result["ci95"] == pytest.approx((1 / 12 - margin, 1 / 12 + margin))
    assert paired_site_difference(first, first, targets) == {
        "difference": 0, "ci95": (0, 0)
    }
    assert paired_site_difference(first[:1], second[:1], targets[:1]) == {
        "difference": 0.5, "ci95": None
    }


def test_scoring_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same shape"):
        site_metrics([[0, 1]], [[0], [1]])
    with pytest.raises(ValueError, match="binary"):
        site_metrics([[0, 2]], [[0, 1]])
    with pytest.raises(ValueError, match="shot and site axes"):
        site_metrics([], [])
    with pytest.raises(ValueError, match="finite"):
        select_threshold([[float("nan"), 0.2]], [[0, 1]])
    with pytest.raises(ValueError, match="same shape"):
        paired_site_difference([[0]], [[0], [1]], [[0]])
