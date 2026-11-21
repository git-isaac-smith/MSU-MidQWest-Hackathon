"""Bounded search over the detector error model's joint faults."""

from dataclasses import asdict, dataclass
from importlib.metadata import version

import numpy as np
import stim

from .decoders import DecoderSpec, compile_decoder


@dataclass(frozen=True)
class HypergraphSpec:
    beam: int = 5
    num_det_orders: int = 2
    pqlimit: int = 5_000
    threads: int = 4

    def __post_init__(self) -> None:
        for value in (self.beam, self.num_det_orders, self.pqlimit, self.threads):
            if type(value) is not int or value < 1:
                raise ValueError("search parameters must be positive integers")

    def to_dict(self) -> dict:
        return {
            "decoder_type": "hypergraph_search",
            "backend": "stabilizer-qec",
            "backend_version": version("stabilizer-qec"),
            "single_shot_threads": 1,
            "det_order_method": "index",
            "seed": 2384753,
            "beam_climbing": False,
            "no_revisit_dets": True,
            "det_penalty": 0.0,
            "merge_errors": True,
            "fallback": "correlated_mwpm",
            **asdict(self),
        }


class HypergraphDecoder:
    """Use correlated matching where the bounded search cannot explain a shot."""

    def __init__(self, model: stim.DetectorErrorModel, spec: HypergraphSpec):
        try:
            from stabilizer_qec import SearchDecoder
        except ImportError as exc:
            raise ImportError("Hypergraph search requires the bascd[hypergraph] extra") from exc

        # Stim's separators annotate a decomposition within one fault. Passing the
        # original DEM to SearchDecoder preserves the joint detector signature.
        self.search = SearchDecoder(
            model, beam=spec.beam, num_det_orders=spec.num_det_orders, pqlimit=spec.pqlimit,
            det_order_method="index", seed=2384753, beam_climbing=False, no_revisit_dets=True,
            det_penalty=0.0, merge_errors=True,
        )
        self.fallback = compile_decoder(model, DecoderSpec("correlated_mwpm"))
        self.fallback.graph.ensure_num_fault_ids(model.num_observables)
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables
        self.threads = spec.threads

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        return self.decode_batch_with_fallback(detectors)[0]

    def decode_batch_with_fallback(self, detectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions, give_up = self.predictions_and_low_confidence(detectors)
        if give_up.any():
            predictions[give_up] = self.fallback.decode_batch(detectors[give_up])
        return predictions, give_up

    def predictions_and_low_confidence(self, detectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return raw search predictions and a flag per unsuccessful shot."""
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError("expected a (shots, detectors) array matching the model")
        if not len(detectors):
            return (np.empty((0, self.num_observables), dtype=np.uint8),
                    np.empty(0, dtype=bool))
        return self.search.decode_batch(
            detectors, return_low_confidence=True, threads=self.threads,
        )

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.ndim != 1 or detectors.shape[0] != self.num_detectors:
            raise ValueError("expected a detector vector matching the model")
        predictions, give_up = self.search.decode_batch(
            detectors[None, :], return_low_confidence=True, threads=1,
        )
        if give_up[0]:
            return self.fallback.decode_one(detectors)
        return predictions[0]


def compile_hypergraph(
    model: stim.DetectorErrorModel, spec: HypergraphSpec = HypergraphSpec(),
) -> HypergraphDecoder:
    """Compile a model with Stim's `decompose_errors=True` for the fallback matcher."""
    return HypergraphDecoder(model, spec)
