"""Run a frozen compute-study decoder or replay its archived accuracy audit."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import stim
from scipy.stats import binomtest

from .compute_campaign import METHODS, digest, factory
from .stats import paired_ler_difference, wilson95


_BLOCK_SHOTS = 4096
_BASELINES = ("ordinary", "activity", "correlated", "belief")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check(condition, location, message):
    if not condition:
        raise ValueError(f"{location}: {message}")


def _frozen_circuit(case, distance=None):
    frozen = _json(case / "freeze.json")
    if distance is not None:
        _check(frozen["distance"] == distance, case, "frozen distance differs from protocol")
    for filename, field in (("circuit.stim", "circuit_sha256"), ("policy.npz", "policy_sha256")):
        _check(digest(case / filename) == frozen[field], case, f"{filename} SHA-256 mismatch")
    _check((case / "policy.npz").stat().st_size == frozen["policy_bytes"], case, "policy size differs from freeze")
    circuit = stim.Circuit.from_file(case / "circuit.stim")
    _check(circuit.num_detectors > 0 and circuit.num_observables == 1, case,
           "expected detector records with one observable")
    return circuit


def _check_detector_padding(detectors, num_detectors, location):
    remainder = num_detectors % 8
    if remainder:
        _check(not np.any(detectors[:, -1] >> remainder), location, "nonzero detector padding")


def _checked_predictions(decode, detectors, location):
    bits = decode(detectors)
    _check(bits.shape == (len(detectors), 1) and bits.dtype == np.uint8 and not np.any(bits > 1),
           location, "decoder did not return one valid prediction byte per shot")
    return bits[:, 0]


def predict(case: Path, method: str, input_path: Path, output_path: Path) -> int:
    """Decode little-endian packed detector records and write one byte per shot."""
    _check(method in METHODS, method, "unknown deployment method")
    circuit = _frozen_circuit(case)
    width = (circuit.num_detectors + 7) // 8
    written = 0
    with input_path.open("rb") as incoming:
        _check(os.fstat(incoming.fileno()).st_size % width == 0, input_path, "partial detector record")
        decode = factory(method, case)
        outgoing = output_path.open("xb")
        try:
            with outgoing:
                while block := incoming.read(_BLOCK_SHOTS * width):
                    _check(len(block) % width == 0, input_path, "partial detector record")
                    detectors = np.frombuffer(block, dtype=np.uint8).reshape(-1, width)
                    _check_detector_padding(detectors, circuit.num_detectors, input_path)
                    bits = _checked_predictions(decode, detectors, method)
                    outgoing.write(bits.tobytes())
                    written += len(bits)
        except BaseException:
            output_path.unlink()
            raise
    return written


def _read_rows(path, fields):
    rows = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in fields)
            _check(key not in rows, path, f"duplicate row {key}")
            rows[key] = row
    return rows


def _verify_row(row, expected, location):
    for field, value in expected.items():
        published = row[field]
        if value is None:
            matches = published == ""
        elif isinstance(value, int):
            matches = int(published) == value
        else:
            matches = math.isclose(float(published), value, rel_tol=1e-10, abs_tol=1e-14)
        _check(matches, location, f"{field} is {published!r}, calculated {value}")


def _audit_archive(path, circuit, seed, n, methods, learned):
    width = (circuit.num_detectors + 7) // 8
    with np.load(path, allow_pickle=False) as archive:
        _check(len(archive.files) == len(methods) + 2 and set(archive.files) == {"detectors", "truth", *methods},
               path, "archive fields differ from protocol")
        detectors, truth = archive["detectors"], archive["truth"]
        _check(detectors.shape == (n, width) and detectors.dtype == np.uint8, path, "invalid detector array")
        _check(truth.shape == (n,) and truth.dtype == np.uint8 and not np.any(truth > 1), path, "invalid truth array")
        _check_detector_padding(detectors, circuit.num_detectors, path)
        fresh_detectors, fresh_truth = circuit.compile_detector_sampler(seed=seed).sample(
            n, separate_observables=True, bit_packed=True)
        _check(np.array_equal(detectors, fresh_detectors), path, "replayed Stim detector bytes differ")
        _check(np.array_equal(truth, fresh_truth[:, 0]), path, "replayed Stim truth bytes differ")
        failures = {}
        for method in methods:
            packed = archive[method]
            _check(packed.shape == ((n + 7) // 8,) and packed.dtype == np.uint8, path, f"invalid {method} predictions")
            if n % 8:
                _check(int(packed[-1]) >> (n % 8) == 0, path, f"nonzero {method} prediction padding")
            bits = np.unpackbits(packed, count=n, bitorder="little")
            if method == "learned" and learned is not None:
                for start in range(0, n, _BLOCK_SHOTS):
                    block = slice(start, start + _BLOCK_SHOTS)
                    replayed = _checked_predictions(learned, detectors[block], path)
                    _check(np.array_equal(bits[block], replayed), path,
                           f"learned deployment differs in block starting at shot {start}")
            failures[method] = bits != truth
    return failures


def audit(run: Path, *, verify_learned: bool = False) -> dict[str, int | bool]:
    """Recreate every Stim record and reconcile archived accuracy and paired statistics."""
    protocol = _json(run / "protocol.json")
    _check(bool(protocol["source_sha256"]), run, "missing source digests")
    for module, expected in protocol["source_sha256"].items():
        _check(module.isidentifier(), run, f"invalid source entry {module}")
        source = Path(__file__).with_name(module + ".py")
        _check(digest(source) == expected, source, "SHA-256 differs from frozen protocol")

    accuracy = _read_rows(run / "accuracy.csv", ("distance", "method"))
    paired = _read_rows(run / "paired.csv", ("distance", "method", "baseline"))
    distances = protocol["distances"]
    methods = (*protocol["methods"], "random_control")
    _check(bool(distances) and len(distances) == len(set(distances)), run, "missing or duplicate distances")
    _check(len(methods) == len(set(methods)) and {*_BASELINES, "learned"} <= set(methods), run, "invalid methods")
    n = protocol["test_shots_per_seed"]
    _check(type(n) is int and n >= 2, run, "each seed requires at least two shots")
    seed_count = 0
    for distance in distances:
        case = run / f"d{distance}"
        circuit = _frozen_circuit(case, distance)
        seeds = protocol["test_seeds"][str(distance)]
        _check(bool(seeds) and len(seeds) == protocol["test_seeds_per_distance"] == len(set(seeds))
               and all(type(seed) is int and 0 <= seed < 2**64 for seed in seeds), case, "invalid test seeds")
        paths = [case / f"test-{seed}.npz" for seed in seeds]
        _check(set(case.glob("test-*.npz")) == set(paths), case, "seed archives differ from protocol")
        learned = factory("learned", case) if verify_learned else None
        failures = [_audit_archive(path, circuit, seed, n, methods, learned)
                    for path, seed in zip(paths, seeds, strict=True)]
        seed_count += len(seeds)
        shots = n * len(seeds)
        for method in methods:
            label = f"{case}/{method}"
            key = (str(distance), method)
            errors = sum(int(np.count_nonzero(row[method])) for row in failures)
            lo, hi = wilson95(errors, shots)
            _verify_row(accuracy.pop(key), dict(shots=shots, errors=errors, error_percent=100 * errors / shots,
                                               wilson_low_percent=100 * lo, wilson_high_percent=100 * hi), label)
            for baseline in _BASELINES:
                wins = [int(np.count_nonzero(row[baseline] & ~row[method])) for row in failures]
                losses = [int(np.count_nonzero(~row[baseline] & row[method])) for row in failures]
                result = paired_ler_difference([n] * len(seeds), wins, losses)
                recovered, introduced = sum(wins), sum(losses)
                total = recovered + introduced
                ci = result["approximate_95_ci"]
                expected = dict(
                    recovered=recovered, introduced=introduced, change_pp=100 * result["ler_difference"],
                    ci_low_pp=100 * ci[0] if ci else None, ci_high_pp=100 * ci[1] if ci else None,
                    paired_p=binomtest(recovered, total, .5).pvalue if total else 1.0,
                )
                _verify_row(paired.pop((*key, baseline)), expected, f"{label} vs {baseline}")
    _check(not accuracy and not paired, run, "unexpected accuracy or paired rows")
    return dict(cases=len(distances), seed_archives=seed_count, shots=seed_count * n,
                accuracy_rows=len(distances) * len(methods), paired_rows=len(distances) * len(methods) * len(_BASELINES),
                learned_replayed=verify_learned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict", help="decode packed detector records using a frozen case")
    prediction.add_argument("--case", type=Path, required=True)
    prediction.add_argument("--method", choices=METHODS, required=True)
    prediction.add_argument("--input", type=Path, required=True, help="one little-endian packed detector record per shot")
    prediction.add_argument("--output", type=Path, required=True, help="one packed observable byte per shot")
    verification = commands.add_parser("audit", help="verify frozen files, Stim records, and aggregate scores")
    verification.add_argument("--run", type=Path, required=True)
    verification.add_argument("--verify-learned", action="store_true", help="also check every deployed learned decision")
    args = parser.parse_args()
    try:
        if args.command == "predict":
            result = {"shots": predict(args.case, args.method, args.input, args.output)}
        else:
            result = audit(args.run, verify_learned=args.verify_learned)
    except (BadZipFile, KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"{args.command} failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
