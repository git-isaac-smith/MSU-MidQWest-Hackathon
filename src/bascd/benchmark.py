import hashlib
import platform
from time import perf_counter

import numpy as np
import pymatching
import stim

from .circuits import memory_circuit
from .stats import paired_counts, shot_failures, wilson95


def run_case(*, basis: str, distance: int, rounds: int, p: float, shots: int, seed: int):
    """Compare the two matchers on a single batch sampled from one circuit."""
    if shots < 1 or not 0 <= seed < 2**64:
        raise ValueError("shots must be positive and seed must be an unsigned 64-bit integer")

    start = perf_counter()
    circuit = memory_circuit(basis, distance, rounds, p)
    model = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    simulation_setup_s = perf_counter() - start
    start = perf_counter()
    detectors, truth = sampler.sample(shots, separate_observables=True)
    sampling_s = perf_counter() - start

    common = dict(
        basis=basis, distance=distance, rounds=rounds, p=p, seed=seed, shots=shots,
        batch_size=shots, noise_model="simple_uniform",
        circuit=f"surface_code:rotated_memory_{basis}",
        circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
        num_detectors=circuit.num_detectors, num_observables=circuit.num_observables,
        simulation_setup_s=simulation_setup_s, sampling_s=sampling_s,
        stim_version=stim.__version__, pymatching_version=pymatching.__version__,
        numpy_version=np.__version__, python_version=platform.python_version(),
        platform=platform.platform(), cpu=platform.processor() or platform.machine(),
        command=(f"python -m bascd --basis {basis} --distance {distance} --rounds {rounds}"
                 f" --p {p} --shots {shots} --seed {seed}"),
    )
    rows = []
    masks = []
    for name, correlated in (("mwpm", False), ("correlated_mwpm", True)):
        start = perf_counter()
        matcher = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
        decoder_setup_s = perf_counter() - start
        start = perf_counter()
        predictions = matcher.decode_batch(detectors, enable_correlations=correlated)
        decode_s = perf_counter() - start
        failed = shot_failures(predictions, truth)
        masks.append(failed)
        failures = int(failed.sum())
        low, high = wilson95(failures, shots)
        rows.append(dict(
            **common, decoder=name, failures=failures, ler=failures / shots,
            wilson95_low=low, wilson95_high=high, decoder_setup_s=decoder_setup_s,
            decode_s=decode_s, batch_us_per_shot=1e6 * decode_s / shots,
        ))

    return {"results": rows, "paired": paired_counts(*masks)}
