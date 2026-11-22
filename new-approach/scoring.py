"""Choose validation thresholds and compare site localization on held-out shots."""

from statistics import NormalDist

import numpy as np


def _shot_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2 or array.size == 0:
        raise ValueError(f"{name} must have nonempty shot and site axes")
    return array


def _binary_shots(values: np.ndarray, name: str) -> np.ndarray:
    array = _shot_array(values, name)
    if array.dtype.kind not in "buif" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain binary values")
    return array.astype(bool, copy=False)


def _check_shape(array: np.ndarray, targets: np.ndarray, name: str) -> None:
    if array.shape != targets.shape:
        raise ValueError(f"{name} and targets must have the same shape")


def select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Maximize pooled positive-site F1 on validation with `probability >= cutoff`.

    Sweep the distinct scores and use the highest cutoff when F1 ties. With
    zero positive validation sites, return 2.0 to select the empty set.
    """
    scores = _shot_array(probabilities, "probabilities")
    labels = _binary_shots(targets, "targets")
    _check_shape(scores, labels, "probabilities")
    if scores.dtype.kind not in "buif" or not np.all((0 <= scores) & (scores <= 1)):
        raise ValueError("probabilities must contain finite values between zero and one")

    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return 2.0

    flat_scores = scores.ravel()
    order = np.argsort(flat_scores)[::-1]
    descending_scores = flat_scores[order]
    ends = np.flatnonzero(np.r_[descending_scores[:-1] != descending_scores[1:], True])
    true_positives = np.cumsum(labels.ravel()[order], dtype=np.int64)[ends]
    f1 = 2 * true_positives / (ends + 1 + positives)
    return float(descending_scores[ends[np.argmax(f1)]])


def site_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, int | float]:
    """Return pooled site counts and rates, plus exact-set accuracy by shot.

    Rates are fractions. Precision, recall, and F1 are zero with a zero
    denominator. Exact-set accuracy is the fraction of fully correct shots.
    """
    actual = _binary_shots(targets, "targets")
    predicted = _binary_shots(predictions, "predictions")
    _check_shape(predicted, actual, "predictions")

    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted)) - tp
    fn = int(np.count_nonzero(actual)) - tp
    tn = actual.size - tp - fp - fn
    shots = actual.shape[0]
    correct = (predicted == actual).reshape(shots, -1).all(axis=1)
    return {
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "shots": shots,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "site_error_rate": (fp + fn) / actual.size,
        "exact_set_accuracy": float(correct.mean()),
    }


def paired_site_difference(
    first: np.ndarray, second: np.ndarray, targets: np.ndarray
) -> dict[str, float | tuple[float, float] | None]:
    """Return the second minus first site error rate and a paired normal 95% CI.

    Difference and interval are in rate units, so 0.01 is one percentage point.
    The standard error uses the between-shot sample standard deviation of the
    paired differences. With one shot, the confidence interval is undefined.
    """
    actual = _binary_shots(targets, "targets")
    first_predictions = _binary_shots(first, "first predictions")
    second_predictions = _binary_shots(second, "second predictions")
    _check_shape(first_predictions, actual, "first predictions")
    _check_shape(second_predictions, actual, "second predictions")

    shots = actual.shape[0]
    sites = actual.size // shots
    first_errors = np.count_nonzero((first_predictions != actual).reshape(shots, sites), axis=1)
    second_errors = np.count_nonzero((second_predictions != actual).reshape(shots, sites), axis=1)
    shot_differences = (second_errors - first_errors) / sites
    difference = float(shot_differences.mean())
    interval = None
    if shots > 1:
        standard_error = float(shot_differences.std(ddof=1) / np.sqrt(shots))
        margin = NormalDist().inv_cdf(0.975) * standard_error
        interval = (difference - margin, difference + margin)
    return {"difference": difference, "ci95": interval}
