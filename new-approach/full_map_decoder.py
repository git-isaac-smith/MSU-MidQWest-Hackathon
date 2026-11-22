"""Logical maximum-a-posteriori inference on every public detector record."""

import json
from pathlib import Path

import numpy as np

from logical_reference import build_reference


MAP_TIE_TOLERANCE = 1e-12


class FullMapDecoder:
    """Integrate fault histories on all shots, including large matching margins."""

    def __init__(self, physical, backend="cached"):
        self.ref = build_reference(*physical)
        if backend == "reference":
            from posterior_decoder import PosteriorDecoder as Engine
        elif backend == "cached":
            from cached_posterior import CachedPosteriorDecoder as Engine
        elif backend == "native":
            from native_posterior import NativePosteriorDecoder as Engine
        else:
            raise ValueError(f"Unknown posterior backend: {backend}")
        self.engine = Engine(*physical)
        self.backend = backend

    @classmethod
    def load(cls, folder, backend="cached"):
        policy = json.loads((Path(folder) / "policy.json").read_text(encoding="utf-8"))
        if policy.get("rounds", 5) != 5:
            raise ValueError("FullMapDecoder requires five noisy check rounds")
        return cls(policy["physical"], backend)

    def probabilities(self, detectors):
        observed, terminal = self.ref.public_checks(detectors)
        return self.engine.predict_proba(observed, terminal)

    def decisions(self, detectors, probabilities):
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (len(detectors),) or not np.isfinite(probabilities).all():
            raise ValueError("Expected one finite logical probability per shot")
        result = probabilities > 0.5
        ties = np.abs(probabilities - 0.5) <= MAP_TIE_TOLERANCE
        if ties.any():
            result[ties] = self.ref.mwpm_parity(detectors[ties])
        return result

    def predict(self, detectors, mode="full_map"):
        if mode != "full_map":
            raise ValueError("FullMapDecoder only implements full_map")
        return self.decisions(detectors, self.probabilities(detectors))
