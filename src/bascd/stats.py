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
