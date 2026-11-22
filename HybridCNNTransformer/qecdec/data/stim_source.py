"""Stim circuit construction, on-the-fly shot sampling, and the MWPM reference decoder.

Verified against stim 1.16 / pymatching 2.4:
  stim.Circuit.generated(code_task, *, distance, rounds, after_clifford_depolarization,
                         before_round_data_depolarization, before_measure_flip_probability,
                         after_reset_flip_probability)
  (note the spelling "depolarization", not "depolarizing").
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import stim

CODE_TASK = "surface_code:rotated_memory_z"


def make_circuit(d: int, p: float, rounds: Optional[int] = None) -> stim.Circuit:
    """Rotated-code Z-memory circuit with uniform circuit-level noise of strength p.

    Noise channels (all with parameter p):
      * DEPOLARIZE1/2 after every Clifford gate,
      * DEPOLARIZE1 on data qubits at the start of every round,
      * X_ERROR before every measurement (measurement flip),
      * X_ERROR after every reset.
    rounds defaults to d.
    """
    return stim.Circuit.generated(
        CODE_TASK,
        distance=d,
        rounds=d if rounds is None else rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


class ShotSource:
    """Seeded stream of (detection events, observable flip) shots for one (d, p).

    sample(n) -> det bool [n, num_detectors], obs bool [n]. Nothing is stored: every call draws
    fresh shots from the compiled Stim detector sampler.
    """

    def __init__(self, d: int, p: float, seed: int, rounds: Optional[int] = None):
        self.d, self.p = d, p
        self.rounds = d if rounds is None else rounds
        self.circuit = make_circuit(d, p, self.rounds)
        self.num_detectors = self.circuit.num_detectors
        self.sampler = self.circuit.compile_detector_sampler(seed=seed)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        det, obs = self.sampler.sample(n, separate_observables=True)
        return det, obs[:, 0]


def build_mwpm(circuit: stim.Circuit):
    """PyMatching decoder from the decomposed detector error model."""
    import pymatching
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def mwpm_predict(matching, det: np.ndarray) -> np.ndarray:
    """Predicted observable flip (bool [N]) for detection events det [N, num_detectors]."""
    pred = matching.decode_batch(det)          # [N, num_observables] uint8
    return np.asarray(pred)[:, 0].astype(bool)
