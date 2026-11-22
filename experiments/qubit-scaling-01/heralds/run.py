"""Run and replay the separate, single-opportunity bath herald benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import pymatching
import stim

from bascd.bath_erasure import BathNoise, LocationDecoder, ZMemory, _paired_log10_p_value
from bascd.stats import paired_ler_difference, wilson95

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "experiments/iteration-03/bath-calibration/bath-profile.json"
DISTANCES = (3, 5, 7, 15, 31)
CLOCKS = ("clock", "selected")
BATCH = 128
SEED = 68904000


def verify_legacy_map(distance):
    code = ZMemory.rotated(distance)
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=1)
    reset = next(i for i, instruction in enumerate(circuit) if instruction.name == "R")
    checks = np.zeros((circuit.num_detectors, len(code.sites)), dtype=np.uint8)
    logical = np.zeros(len(code.sites), dtype=np.uint8)
    for index, site in enumerate(code.sites):
        injected = stim.Circuit()
        injected.append("X_ERROR", [site], .125)
        faulted = circuit[:reset + 1] + injected + circuit[reset + 1:]
        error, = [op for op in faulted.detector_error_model() if op.type == "error"]
        for target in error.targets_copy():
            if target.is_relative_detector_id():
                checks[target.val, index] = 1
            elif target.is_logical_observable_id() and target.val == 0:
                logical[index] = 1
            else:
                raise ValueError("unexpected injected fault target")
    np.testing.assert_array_equal(code.checks, checks[np.any(checks, axis=1)])
    np.testing.assert_array_equal(code.logical_z, logical)


def evaluate(code, noise, shots, seed, *, timed):
    paulis, flags = noise.sample_paulis(shots, len(code.sites), np.random.default_rng(seed))
    syndrome, truth = code.measure(paulis)
    decoder = LocationDecoder(code, noise)
    methods = {"blind": lambda sl: decoder.decode_blind(syndrome[sl]),
               "heralded": lambda sl: decoder.decode_heralded(syndrome[sl], flags[sl])}
    predictions = {method: np.empty_like(truth) for method in methods}
    times = {method: [] for method in methods}
    order = np.random.default_rng(seed + 99000)
    if timed:
        for decode in methods.values():
            decode(slice(0, BATCH))
    for begin in range(0, shots, BATCH):
        stop = min(begin + BATCH, shots)
        chunk = slice(begin, stop)
        for name in order.permutation(list(methods)):
            start = perf_counter_ns()
            prediction = methods[name](chunk)
            elapsed = perf_counter_ns() - start
            predictions[name][chunk] = prediction
            if timed and stop - begin == BATCH:
                times[name].append(elapsed / (1000 * BATCH))
    return dict(syndrome=syndrome, flags=flags, truth=truth, **predictions), times


def failure_stats(payload, shots):
    blind, heralded = (payload[name] != payload["truth"] for name in ("blind", "heralded"))
    recovered = int((blind & ~heralded).sum())
    regressed = int((~blind & heralded).sum())
    paired = paired_ler_difference([shots], [recovered], [regressed])
    paired_interval = paired["approximate_95_ci"]
    row = dict(recovered=recovered, regressed=regressed, heralded_minus_blind_pp=100 * paired["ler_difference"],
               paired95_low_pp=100 * paired_interval[0] if paired_interval else "",
               paired95_high_pp=100 * paired_interval[1] if paired_interval else "",
               mcnemar_log10_p=_paired_log10_p_value(recovered, regressed))
    for name, failures in (("blind", blind), ("heralded", heralded)):
        count = int(failures.sum())
        low, high = wilson95(count, shots)
        row.update({f"{name}_failures": count, f"{name}_ler_pct": 100 * count / shots,
                    f"{name}_wilson95_low_pct": 100 * low, f"{name}_wilson95_high_pct": 100 * high})
    return row


def run(folder):
    profile_bytes = PROFILE.read_bytes()
    profile = json.loads(profile_bytes)
    for distance in (3, 5):
        verify_legacy_map(distance)
    folder.mkdir(parents=True, exist_ok=False)
    rows, cases = [], []
    for distance in DISTANCES:
        code = ZMemory.rotated(distance)
        shots = 4096 if distance <= 7 else (2048 if distance == 15 else 1024)
        for clock_i, clock in enumerate(CLOCKS):
            seed = SEED + 10 * distance + clock_i
            noise = BathNoise.from_profile_entry(profile[clock])
            payload, times = evaluate(code, noise, shots, seed, timed=True)
            archive = f"d{distance}-{clock}.npz"
            archived = {name: np.packbits(values, axis=1, bitorder="little") for name, values in payload.items()}
            np.savez_compressed(folder / archive, **archived)
            row = dict(distance=distance, data_qubits=distance**2, syndrome_bits=len(code.checks),
                       clock=clock, time=noise.time, shots=shots, seed=seed,
                       erasure_probability=noise.erasure_probability, pass_flip_probability=noise.passed_flip_probability,
                       marginal_flip_probability=noise.blind_flip_probability, flags=int(payload["flags"].sum()),
                       **failure_stats(payload, shots))
            for name, samples in times.items():
                row[f"{name}_us_per_shot"] = median(samples)
                row[f"{name}_batch_us_per_shot"] = json.dumps(samples)
            rows.append(row)
            cases.append(dict(distance=distance, clock=clock, shots=shots, seed=seed, archive=archive))
            print(f"d{distance} {clock}: {shots} paired shots", flush=True)
    with (folder / "accuracy-timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(cases=cases, versions={module.__name__: module.__version__ for module in (np, pymatching, stim)},
                    profile_sha256=hashlib.sha256(profile_bytes).hexdigest(), batch_shots=BATCH,
                    timing="Median inference time per shot using the same interleaved chunks. Setup, sampling and scoring excluded.",
                    archive="All arrays are little-endian packed along axis 1. True logical parity is evaluator-only.",
                    graph_verify="Detecting regions at tick zero matched one possible X_ERROR per site at d3 and d5.")
    (folder / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def verify(folder):
    metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    profile_bytes = PROFILE.read_bytes()
    if hashlib.sha256(profile_bytes).hexdigest() != metadata["profile_sha256"]:
        raise ValueError("Bath profile differs from archive")
    profile = json.loads(profile_bytes)
    with (folder / "accuracy-timing.csv").open(newline="", encoding="utf-8") as stream:
        rows = {(int(row["distance"]), row["clock"]): row for row in csv.DictReader(stream)}
    for distance in (3, 5):
        verify_legacy_map(distance)
    for case in metadata["cases"]:
        distance, clock, shots = case["distance"], case["clock"], case["shots"]
        noise = BathNoise.from_profile_entry(profile[clock])
        payload, _ = evaluate(ZMemory.rotated(distance), noise, shots, case["seed"], timed=False)
        with np.load(folder / case["archive"], allow_pickle=False) as archive:
            for name, values in payload.items():
                np.testing.assert_array_equal(np.packbits(values, axis=1, bitorder="little"), archive[name])
        row = rows[distance, clock]
        for name, result in failure_stats(payload, shots).items():
            if str(result) != row[name]:
                raise ValueError(f"d{distance} {clock}: incorrect {name}")
        print(f"d{distance} {clock}: verified samples, predictions and scores", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    (run if args.action == "run" else verify)(args.out)


if __name__ == "__main__":
    main()
