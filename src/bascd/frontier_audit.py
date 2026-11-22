"""Independently verify circuit-frontier shot archives against published metrics."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.stats import binomtest


def _check(condition: bool, location: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{location}: {message}")


def _verify_number(row: dict, field: str, calculated: float, location: str) -> None:
    published = float(row[field])
    _check(math.isclose(published, calculated, rel_tol=1e-10, abs_tol=1e-14),
           location, f"{field} is {published}, calculated {calculated}")


def _wilson(errors: int, shots: int) -> tuple[float, float]:
    z = 1.9599639845400536
    p = errors / shots
    center = p + z**2 / (2 * shots)
    radius = z * math.sqrt(p * (1 - p) / shots + z**2 / (4 * shots**2))
    divisor = 1 + z**2 / shots
    lower = (center - radius) / divisor if errors else 0.
    upper = (center + radius) / divisor if errors < shots else 1.
    return lower, upper


def _read_case_rows(csv_path: Path) -> dict[str, dict[str, dict]]:
    cases = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            methods = cases.setdefault(row["case"], {})
            _check(row["method"] not in methods, row["case"], f"duplicate {row['method']} metrics")
            methods[row["method"]] = row
    _check(bool(cases), str(csv_path), "missing metrics")
    return cases


def _audit_case(
    case_dir: Path, rows: dict[str, dict], resample: bool, planned: dict | None,
) -> tuple[int, int]:
    location = case_dir.name
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    _check(manifest["case"] == location, location, "manifest case differs from archive directory")
    if planned is not None:
        for field, value in planned.items():
            _check(manifest.get(field) == value, location, f"manifest {field} differs from protocol")
    integer_fields = ("distance", "rounds")
    text_fields = ("basis", "variant", "study")
    for method, row in rows.items():
        for field in integer_fields + text_fields + ("p",):
            _check(str(row[field]) == str(manifest[field]) or (
                field in integer_fields + ("p",) and float(row[field]) == float(manifest[field])
            ), location, f"{method} {field} differs from manifest")

    circuit = stim.Circuit.from_file(case_dir / "circuit.stim")
    _check(circuit.num_observables == 1, location, "expected exactly one observable")
    _check("mwpm" in rows, location, "missing ordinary MWPM baseline")
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True),
    )
    shots = manifest["shots_per_seed"]
    seeds = manifest["seeds"]
    _check(type(shots) is int and shots > 0, location, "invalid shot count")
    _check(bool(seeds) and len(seeds) == len(set(seeds)), location, "missing or duplicate seeds")
    expected_paths = {case_dir / f"seed-{seed}.npz" for seed in seeds}
    _check(set(case_dir.glob("seed-*.npz")) == expected_paths, location,
           "seed archives differ from manifest")
    counts = {method: np.zeros(3, dtype=np.int64) for method in rows}
    detector_width = (circuit.num_detectors + 7) // 8
    packed_prediction_length = (shots + 7) // 8
    for seed in seeds:
        label = f"{location}/seed-{seed}"
        with np.load(case_dir / f"seed-{seed}.npz", allow_pickle=False) as data:
            _check(set(data.files) == {"detectors", "truth", *rows}, label, "archive fields differ from metrics")
            detectors, truth = data["detectors"], data["truth"]
            _check(detectors.shape == (shots, detector_width) and detectors.dtype == np.uint8,
                   label, "invalid detector array")
            _check(truth.shape == (shots, 1) and truth.dtype == np.uint8 and np.all(truth <= 1),
                   label, "invalid packed truth")
            remainder = circuit.num_detectors % 8
            if remainder:
                _check(np.all((detectors[:, -1] >> remainder) == 0), label, "nonzero detector padding")
            if resample:
                fresh_detectors, fresh_truth = circuit.compile_detector_sampler(seed=seed).sample(
                    shots, separate_observables=True, bit_packed=True,
                )
                _check(np.array_equal(detectors, fresh_detectors), label, "Stim detectors differ")
                _check(np.array_equal(truth, fresh_truth), label, "Stim observables differ")
            replay_mwpm = matcher.decode_batch(detectors, bit_packed_shots=True, bit_packed_predictions=True)
            replay_packed = np.packbits(replay_mwpm[:, 0], bitorder="little")
            _check(np.array_equal(data["mwpm"], replay_packed), label, "ordinary MWPM predictions differ")
            baseline_failure = replay_mwpm[:, 0] != truth[:, 0]
            for method in counts:
                packed = data[method]
                _check(packed.shape == (packed_prediction_length,) and packed.dtype == np.uint8,
                       label, f"invalid {method} predictions")
                if shots % 8:
                    _check(int(packed[-1]) >> (shots % 8) == 0, label, f"nonzero {method} padding")
                failed = np.unpackbits(packed, count=shots, bitorder="little") != truth[:, 0]
                counts[method] += (
                    failed.sum(),
                    (baseline_failure & ~failed).sum(),
                    (~baseline_failure & failed).sum(),
                )

    pooled_shots = shots * len(seeds)
    for method, (errors, fixed, added) in counts.items():
        row = rows[method]
        label = f"{location}/{method}"
        for field, calculated in (
            ("shots", pooled_shots), ("errors", errors), ("fixed", fixed), ("added", added),
        ):
            _check(int(row[field]) == calculated, label, f"{field} is {row[field]}, calculated {calculated}")
        lower, upper = _wilson(int(errors), pooled_shots)
        paired_p = binomtest(int(fixed), int(fixed + added)).pvalue if fixed + added else 1.0
        batch = np.asarray(manifest["timings"][method], dtype=float)
        singles = np.asarray(manifest["single_us"][method], dtype=float)
        _check(batch.ndim == 1 and batch.size and np.all(np.isfinite(batch)) and np.all(batch > 0),
               label, "invalid per-batch timings")
        _check(singles.ndim == 1 and singles.size and np.all(np.isfinite(singles)) and np.all(singles > 0),
               label, "invalid single-shot timings")
        _check(batch.size == manifest.get("timing_repeats", batch.size), label, "missing batch timings")
        _check(singles.size == manifest.get("latency_shots", singles.size), label, "missing single-shot timings")
        batch_us_shot = np.median(batch)
        for field, calculated in (
            ("ler_percent", errors / pooled_shots * 100),
            ("wilson_low", lower * 100), ("wilson_high", upper * 100),
            ("paired_p", paired_p), ("batch_us_shot", batch_us_shot),
            ("batch_us_round", batch_us_shot / manifest["rounds"]),
            ("single_median_us", np.median(singles)), ("single_p95_us", np.percentile(singles, 95)),
            ("setup_s", manifest["setup_s"][method]),
        ):
            _verify_number(row, field, calculated, label)
    return len(seeds), pooled_shots


def audit(run: Path, *, resample: bool = False) -> dict[str, int | bool]:
    """Replay ordinary MWPM, optionally replay Stim, and reconcile every metrics row."""
    cases = _read_case_rows(run / "metrics.csv")
    archived = {directory.name for directory in (run / "cases").iterdir() if directory.is_dir()}
    _check(set(cases) == archived, str(run), "metrics cases differ from archived cases")
    planned_cases = {}
    protocol_path = run / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        planned_cases = {case["case"]: case for case in protocol["cases"]}
        _check(len(planned_cases) == len(protocol["cases"]) and set(planned_cases) == archived,
               str(run), "archived cases differ from protocol")
        for case, rows in cases.items():
            _check(set(rows) == set(protocol["methods"]), case, "metrics methods differ from protocol")
    seeds = shots = 0
    for case, rows in cases.items():
        case_seeds, case_shots = _audit_case(run / "cases" / case, rows, resample, planned_cases.get(case))
        seeds += case_seeds
        shots += case_shots
    return {
        "cases": len(cases), "metrics_rows": sum(len(rows) for rows in cases.values()),
        "seed_archives": seeds, "shots": shots, "stim_replayed": resample, "mwpm_replayed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resample", action="store_true", help="Replay and compare all Stim detector and truth bytes.")
    args = parser.parse_args()
    try:
        result = audit(args.run, resample=args.resample)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
