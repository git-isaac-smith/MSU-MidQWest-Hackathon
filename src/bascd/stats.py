from math import sqrt
from statistics import NormalDist

import numpy as np


def shot_failures(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Count each shot with one or more mismatched observables once."""
    if truth.ndim != 2 or predictions.shape != truth.shape or truth.shape[1] == 0:
        raise ValueError("expected matching (shots, observables) arrays")
    return np.any(predictions != truth, axis=1)


def wilson95(failures: int, shots: int) -> tuple[float, float]:
    if not 0 <= failures <= shots or shots <= 0:
        raise ValueError("expected a positive shot count and 0 <= failures <= shots")
    z = NormalDist().inv_cdf(0.975)
    fraction = failures / shots
    divisor = 1 + z**2 / shots
    center = (fraction + z**2 / (2 * shots)) / divisor
    radius = z * sqrt(fraction * (1 - fraction) / shots + z**2 / (4 * shots**2)) / divisor
    return (0.0 if failures == 0 else center - radius,
            1.0 if failures == shots else center + radius)


def paired_counts(baseline: np.ndarray, correlated: np.ndarray) -> dict[str, int]:
    if baseline.ndim != 1 or baseline.shape != correlated.shape:
        raise ValueError("expected paired failure masks of equal length")
    if baseline.dtype != np.bool_ or correlated.dtype != np.bool_:
        raise ValueError("expected boolean failure masks")
    return dict(
        both_correct=int((~baseline & ~correlated).sum()),
        mwpm_only_failed=int((baseline & ~correlated).sum()),
        correlated_only_failed=int((~baseline & correlated).sum()),
        both_failed=int((baseline & correlated).sum()),
    )


def paired_ler_difference(shots, recovered, lost):
    """Candidate minus reference LER on common shots in independent scenarios."""
    strata = list(zip(shots, recovered, lost, strict=True))
    if not strata or any(n < 2 or min(w, l) < 0 or w + l > n for n, w, l in strata):
        raise ValueError("each paired scenario needs valid discordant counts and at least two shots")
    total = sum(n for n, _, _ in strata)
    wins = sum(w for _, w, _ in strata)
    losses = sum(l for _, _, l in strata)
    delta = (losses - wins) / total
    variance = sum((w + l - (l - w)**2 / n) * n / (n - 1) for n, w, l in strata) / total**2
    margin = 1.96 * sqrt(variance)
    return dict(recovered=wins, lost=losses, ler_difference=delta,
                approximate_95_ci=[delta - margin, delta + margin] if wins + losses >= 30 else None)
