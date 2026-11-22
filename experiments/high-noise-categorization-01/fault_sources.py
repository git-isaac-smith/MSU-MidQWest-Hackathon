"""Measure the effect of removing one physical fault family from the same Stim shot."""

import argparse
import hashlib
from collections import defaultdict
from math import log1p
from pathlib import Path

import numpy as np
import pymatching
import stim

from bascd.circuits import memory_circuit
from bascd.decoders import DecoderSpec, compile_decoder
from bascd.records import save_csv, save_json
from bascd.stats import paired_ler_difference, shot_failures, wilson95


FAMILIES = {
    "gate": "after_clifford_depolarization",
    "round_start_data": "before_round_data_depolarization",
    "readout": "before_measure_flip_probability",
    "reset": "after_reset_flip_probability",
}
LEVELS = (0.005, 0.01, 0.02)
SEED = 724096830
SHOTS = 20_000


def sample(circuit, seed):
    return circuit.compile_detector_sampler(seed=seed).sample(SHOTS, separate_observables=True)


def signature_log_weights(circuits):
    """Accumulate log(1 - 2p) for each independent detector/observable parity flip."""
    weights = defaultdict(float)
    for circuit in circuits:
        for term in circuit.detector_error_model().flattened():
            if term.type == "error":
                signature = tuple(sorted(str(target) for target in term.targets_copy()))
                weights[signature] += log1p(-2 * term.args_copy()[0])
    return weights


def run_case(p, case_index):
    full = memory_circuit("z", 7, 7, p)
    decoder = compile_decoder(full.detector_error_model(decompose_errors=True), DecoderSpec("correlated_mwpm"))
    effects, seeds, sources = {}, {}, []
    detectors = np.zeros((SHOTS, full.num_detectors), dtype=bool)
    truth = np.zeros((SHOTS, full.num_observables), dtype=bool)
    for i, (family, argument) in enumerate(FAMILIES.items()):
        circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=7, rounds=7, **{argument: p})
        if circuit.without_noise() != full.without_noise():
            raise ValueError("Circuit layout changed between fault families")
        sources.append(circuit)
        seeds[family] = SEED + 100 * case_index + i
        effects[family] = sample(circuit, seeds[family])
        detectors ^= effects[family][0]
        truth ^= effects[family][1]

    native = signature_log_weights([full])
    composed = signature_log_weights(sources)
    if native.keys() != composed.keys():
        raise ValueError("Fault families have different joint detector/observable signatures")
    max_delta = max(abs(weight - composed[key]) for key, weight in native.items())
    if max_delta > 1e-12:
        raise ValueError("Fault families do not reproduce the full model's parity probabilities")

    mixed = shot_failures(decoder.decode_batch(detectors), truth)
    n_failed = int(mixed.sum())
    rows = []
    rescues = np.zeros(SHOTS, dtype=int)
    for name, (family_detectors, family_truth) in effects.items():
        stripped = shot_failures(decoder.decode_batch(detectors ^ family_detectors), truth ^ family_truth)
        recovered = mixed & ~stripped
        rescues += recovered
        wins, losses = int(recovered.sum()), int((~mixed & stripped).sum())
        paired = paired_ler_difference([SHOTS], [wins], [losses])
        rows.append(dict(p=p, removed=name, shots=SHOTS, full_failures=n_failed,
                         remaining_failures=int(stripped.sum()), recovered=wins, introduced=losses,
                         net_reduction_pp=100 * (wins - losses) / SHOTS,
                         reduction_ci95_pp=[-100 * v for v in reversed(paired["approximate_95_ci"])]
                         if paired["approximate_95_ci"] else None))
        if n_failed - wins + losses != int(stripped.sum()):
            raise ValueError("Paired counts did not reconcile")

    # An independent sample from the ordinary all-noise generator checks the component construction.
    direct_seed = SEED + 100 * case_index + len(FAMILIES)
    direct_detectors, direct_truth = sample(full, direct_seed)
    direct_failed = int(shot_failures(decoder.decode_batch(direct_detectors), direct_truth).sum())
    return rows, dict(p=p, shots=SHOTS, full_failures=n_failed, full_wilson95=wilson95(n_failed, SHOTS),
                      direct_check_failures=direct_failed, direct_check_wilson95=wilson95(direct_failed, SHOTS),
                      direct_check_seed=direct_seed, seeds=seeds,
                      verified_joint_signatures=len(native), max_abs_log_weight_difference=max_delta,
                      circuit_sha256=hashlib.sha256(str(full).encode()).hexdigest(),
                      failed_shots_rescued_by_n_removals=np.bincount(rescues[mixed], minlength=5).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="New output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, cases = [], []
    for i, p in enumerate(LEVELS):
        group, case = run_case(p, i)
        rows.extend(group)
        cases.append(case)
    save_csv(args.out / "removals.csv", rows)
    save_json(args.out / "summary.json", dict(
        noise_model="provisional simple_uniform", distance=7, basis="z", rounds=7, seed=SEED,
        decoder="Native correlated matching compiled once per level from the all-noise circuit",
        families=FAMILIES, shots_per_case=SHOTS, stim=stim.__version__, pymatching=pymatching.__version__,
        numpy=np.__version__, cases=cases))


if __name__ == "__main__":
    main()
