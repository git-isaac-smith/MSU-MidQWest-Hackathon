"""Sampling, timed inference and paired scoring for Iteration 02."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import stim

from .circuits import memory_circuit
from .decoders import DecoderSpec, compile_decoder
from .stats import shot_failures, wilson95

PROBABILITIES = (0.001, 0.003, 0.005, 0.01)
SPLIT_TAGS = {"sweep": 0, "train": 1, "validation": 2, "final_test": 3}


@dataclass(frozen=True)
class Scenario:
    distance: int
    p: float
    basis: str

    def seed(self, root: int, split: str) -> int:
        if type(root) is not int or not 0 <= root < 2**64:
            raise ValueError("root seed must be an unsigned 64-bit integer")
        key = json.dumps([root, self.distance, self.p, self.basis], separators=(",", ":"))
        digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        return (SPLIT_TAGS[split] << 62) | (digest >> 2)


@dataclass
class Samples:
    model: stim.DetectorErrorModel
    detectors: np.ndarray
    truth: np.ndarray
    metadata: dict


def sample(case: Scenario, root: int, split: str, shots: int) -> Samples:
    if split not in ("sweep", "train", "validation"):
        raise ValueError("final-test samples are reserved for the frozen comparison")
    return sample_circuit(lambda: memory_circuit(case.basis, case.distance, case.distance, case.p),
                          case.seed(root, split), split, shots,
                          dict(**asdict(case), rounds=case.distance, noise_model="simple_uniform",
                               noise_status="provisional"))


def sample_circuit(build_circuit, seed: int, split: str, shots: int, scenario: dict) -> Samples:
    if shots < 1:
        raise ValueError("shots must be positive")
    start = perf_counter_ns()
    circuit = build_circuit()
    model = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    setup_s = (perf_counter_ns() - start) / 1e9
    start = perf_counter_ns()
    detectors, truth = sampler.sample(shots, separate_observables=True)
    sampling_s = (perf_counter_ns() - start) / 1e9
    detectors.flags.writeable = truth.flags.writeable = False
    meta = dict(**scenario, shots=shots, batch_size=shots, split=split, seed=seed,
                circuit_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
                sample_sha256=hashlib.sha256(detectors.tobytes() + truth.tobytes()).hexdigest(),
                num_detectors=circuit.num_detectors, num_observables=circuit.num_observables,
                simulation_setup_s=setup_s, sampling_s=sampling_s)
    return Samples(model, detectors, truth, meta)


def measure(data: Samples, spec: DecoderSpec, repeats: int, latency_shots: int = 0):
    return measure_decoder(data, lambda: compile_decoder(data.model, spec), spec.decoder_type,
                           spec.to_dict(), repeats, latency_shots)


def measure_decoder(data, compile_matcher, name, settings, repeats, latency_shots=0):
    if repeats < 1 or not 0 <= latency_shots <= len(data.detectors):
        raise ValueError("invalid timing repetition count or latency sample size")
    start = perf_counter_ns()
    decoder = compile_matcher()
    decoder.decode_batch(np.zeros((1, data.model.num_detectors), dtype=bool))
    setup_s = (perf_counter_ns() - start) / 1e9
    batch_s = []
    for _ in range(repeats):
        start = perf_counter_ns()
        predictions = decoder.decode_batch(data.detectors)
        batch_s.append((perf_counter_ns() - start) / 1e9)
    decode_s = float(np.median(batch_s))
    latency_us = []
    for index, event in enumerate(data.detectors[:latency_shots]):
        start = perf_counter_ns()
        single = decoder.decode_one(event)
        latency_us.append((perf_counter_ns() - start) / 1e3)
        if not np.array_equal(single, predictions[index]):
            raise ValueError("single-shot and batch predictions disagree")
    failed = shot_failures(predictions, data.truth)
    count, shots = int(failed.sum()), len(failed)
    low, high = wilson95(count, shots)
    row = dict(**data.metadata, settings=settings, decoder=name,
               failures=count, ler=count / shots, wilson95_low=low, wilson95_high=high,
               decoder_setup_s=setup_s, batch_times_s=batch_s, median_batch_s=decode_s,
               batch_us_per_shot=1e6 * decode_s / shots, shots_per_second=shots / decode_s,
               latency_shots=latency_shots,
               single_p50_us=float(np.median(latency_us)) if latency_us else None,
               single_p95_us=float(np.percentile(latency_us, 95)) if latency_us else None)
    return row, failed


def environment():
    package = Path(__file__).parent
    sources = sorted([*package.glob("*.py"), *package.glob("schemas/*.json")])
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "versions": {p: version(p) for p in ("bascd", "numpy", "stim", "pymatching", "scipy",
                                              "matplotlib", "jsonschema", "filelock")},
        "source_sha256": {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }
