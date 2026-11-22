"""Inference-only five-round logical decoder. All inputs are public detector bits."""

import json
from pathlib import Path

import numpy as np

from frequency import Passband, features
from hybrid_cnn import HybridCNN, HybridFeatures, make_features
from localizer import Localizer
from logical_reference import build_reference
from matching_confidence import MatchingConfidence
from posterior_decoder import PosteriorDecoder


def gap(costs):
    return np.abs(costs[:, 1] - costs[:, 0])


def gap_priority(detectors, costs):
    """Prioritize low margin. A fixed syndrome hash orders equal margins."""
    key = np.full(len(detectors), 2166136261, dtype=np.uint32)
    for column in np.packbits(detectors, axis=1).T:
        key = (key ^ column.astype(np.uint32)) * np.uint32(16777619)
    return -np.round(gap(costs), 5) + (key.astype(np.float64) / 2**32) * 1e-6


def raw_ablation(x):
    """Remove Fourier fields, pretrained site probabilities and their derived scalar."""
    spatial = x.spatial.copy()
    spatial[..., 5:11] = 0  # channel 11 is the public check geometry
    spatial[..., 18] = 0
    matching = x.matching.copy()
    matching[:, -1] = 0
    return HybridFeatures(spatial, matching, x.check_mask)


def decision_rule(score, override_threshold):
    return score > override_threshold


def cnn_priority(score):
    # A teacher probability near 1/2 leaves the learned decision most vulnerable to approximation error.
    return -np.abs(score - 0.5)


class HybridDecoder:
    """Load a trained policy and accept the Stim circuit's unmodified detector records."""

    def __init__(self, physical, site_model, site_threshold, band):
        self.ref = build_reference(*physical)
        self.confidence = MatchingConfidence(*physical)
        self.posterior = PosteriorDecoder(*physical)
        self.site_model = site_model
        self.site_threshold = site_threshold
        self.band = band
        self.network = None
        self.policy = None

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        spec = json.loads((folder / "policy.json").read_text(encoding="utf-8"))
        obj = cls(spec["physical"], Localizer.load(folder / "site-cnn.npz"),
                  spec["site_threshold"], Passband(**spec["passband"]))
        obj.network = HybridCNN.load(folder / "logical-cnn.npz")
        obj.policy = spec
        return obj

    def encode(self, detectors, baseline, costs):
        observed, terminal = self.ref.public_checks(detectors)
        site_scores = self.site_model.predict(features(observed, self.ref.check_mask, self.band))
        legacy = self.ref.observable_parity(self.ref.complete_fft(site_scores >= self.site_threshold, terminal))
        return make_features(observed, terminal, self.ref.check_mask, baseline, costs[:, 1] - costs[:, 0],
                             mwpm_weight=costs.min(axis=1), fft_site_probabilities=site_scores,
                             matching_features=(legacy != baseline)[:, None], passband=self.band)

    def exact_decisions(self, detectors, baseline):
        observed, terminal = self.ref.public_checks(detectors)
        p = self.posterior.predict_proba(observed, terminal)
        return np.where(np.abs(p - 0.5) <= 1e-12, baseline, p > 0.5)

    def predict(self, detectors, mode="balanced"):
        """Return logical flips for a batch. Modes are learned, balanced, accuracy, and margin_only."""
        if self.network is None or self.policy is None:
            raise ValueError("Load a trained hybrid policy before inference")
        if mode not in ("learned", "balanced", "accuracy", "margin_only"):
            raise ValueError(f"Unknown decoding mode {mode}")
        baseline = self.ref.mwpm_parity(detectors)
        if not len(detectors):
            return baseline
        costs = self.confidence.features(detectors)
        idx = np.flatnonzero(gap(costs) <= self.policy["gap_limit"])
        if not len(idx):
            return baseline
        records = detectors[idx]
        result = baseline.copy()
        if mode == "accuracy":
            expert = np.ones(len(idx), dtype=bool)
        elif mode == "margin_only":
            expert = gap_priority(records, costs[idx]) >= self.policy["gap_route_at"]
        else:
            score = self.network.predict(self.encode(records, baseline[idx], costs[idx]))
            result[idx] ^= decision_rule(score, self.policy["override_threshold"])
            expert = (cnn_priority(score) >= self.policy["cnn_route_at"]) if mode == "balanced" else np.zeros(len(idx), bool)
        if expert.any():
            routed = idx[expert]
            result[routed] = self.exact_decisions(detectors[routed], baseline[routed])
        return result
