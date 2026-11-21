"""Verify a Stim shot archive against the stored circuits and count tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import stim


DECODERS = ("native", "bath_clock", "frozen")
REFERENCES = ("native", "bath_clock")
BIT_COUNTS = np.array([byte.bit_count() for byte in range(256)], dtype=np.uint8)


def read_table(
    path: Path, key_columns: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = tuple(row[column] for column in key_columns)
            if key in rows:
                raise ValueError(f"{path.name}: duplicate row {key}")
            rows[key] = row
    return rows


def read_shots(
    path: Path, *, shots: int, bits: int, is_detector: bool = False
) -> np.ndarray:
    width = (bits + 7) // 8
    expected_size = shots * width
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path.name}: {actual_size} bytes, expected {expected_size}")
    if not width:
        return np.empty((shots, 0), dtype=np.uint8)
    dimension = "num_detectors" if is_detector else "num_observables"
    data = stim.read_shot_data_file(
        path=path, format="b8", bit_packed=True, **{dimension: bits}
    )
    if data.shape != (shots, width):
        raise ValueError(f"{path.name}: read shape {data.shape}, expected {(shots, width)}")
    if bits % 8 and np.any(data[:, -1] >> (bits % 8)):
        raise ValueError(f"{path.name}: nonzero padding bits")
    return data


def check_counts(row: dict[str, str], expected: dict[str, int], label: str) -> None:
    for column, count in expected.items():
        value = int(row[column])
        if value != count:
            raise ValueError(f"{label}: {column} is {value}, expected {count}")


def verify_case(
    folder: Path,
    stats: dict[str, dict[str, str]],
    paired: dict[str, dict[str, str]],
    *,
    shots: int,
    seed: int,
    resample: bool,
) -> None:
    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    detectors = read_shots(
        folder / "detectors.b8", shots=shots, bits=circuit.num_detectors, is_detector=True
    )
    observables = read_shots(
        folder / "observables.b8", shots=shots, bits=circuit.num_observables
    )
    sample_counts = {
        "seed": seed,
        "shots": shots,
        "detector_events": int(BIT_COUNTS[detectors].sum()),
        "shots_with_events": np.count_nonzero(np.any(detectors, axis=1)),
        "observable_flips": int(BIT_COUNTS[observables].sum()),
    }
    failures = {}
    for decoder in DECODERS:
        predictions = read_shots(
            folder / f"{decoder}.b8", shots=shots, bits=circuit.num_observables
        )
        failed = np.any(predictions != observables, axis=1)
        failures[decoder] = failed
        check_counts(
            stats[decoder],
            dict(sample_counts, failures=np.count_nonzero(failed)),
            f"stats.csv {decoder}",
        )

    candidate_failed = failures["frozen"]
    for reference, row in paired.items():
        reference_failed = failures[reference]
        check_counts(
            row,
            {
                "recovered": np.count_nonzero(reference_failed & ~candidate_failed),
                "lost": np.count_nonzero(~reference_failed & candidate_failed),
                "both_failed": np.count_nonzero(reference_failed & candidate_failed),
                "both_correct": np.count_nonzero(~reference_failed & ~candidate_failed),
            },
            f"paired.csv {reference}",
        )

    if resample:
        sampled_detectors, sampled_observables = circuit.compile_detector_sampler(
            seed=seed
        ).sample(shots, separate_observables=True, bit_packed=True)
        if not np.array_equal(detectors, sampled_detectors):
            raise ValueError("detectors.b8 differs from the seeded Stim sample")
        if not np.array_equal(observables, sampled_observables):
            raise ValueError("observables.b8 differs from the seeded Stim sample")


def verify_archive(folder: Path, *, resample: bool) -> tuple[int, int]:
    with (folder / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("manifest.json must specify format_version 1")
    shots = manifest["shots_per_scenario"]
    if type(shots) is not int or shots <= 0:
        raise ValueError("manifest.json shots_per_scenario must be a positive integer")
    seeds = {}
    for row in manifest["cases"]:
        case, seed = row["case"], row["seed"]
        if not isinstance(case, str) or not case or case in seeds:
            raise ValueError(f"manifest.json has an invalid or duplicate case name {case!r}")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError(f"manifest.json case {case} has an invalid seed")
        seeds[case] = seed
    if not seeds:
        raise ValueError("manifest.json lists no cases")

    stats = read_table(folder / "stats.csv", ("case", "decoder"))
    paired = read_table(folder / "paired.csv", ("case", "reference"))
    cases = {path.name: path for path in (folder / "cases").iterdir() if path.is_dir()}
    if cases.keys() != seeds.keys():
        raise ValueError(
            f"cases: missing directories {sorted(seeds.keys() - cases.keys())}, "
            f"unexpected directories {sorted(cases.keys() - seeds.keys())}"
        )
    expected_stats = {(case, decoder) for case in cases for decoder in DECODERS}
    expected_pairs = {
        (case, reference) for case in cases for reference in REFERENCES
    }
    for filename, expected, actual in (
        ("stats.csv", expected_stats, stats.keys()),
        ("paired.csv", expected_pairs, paired.keys()),
    ):
        if expected != actual:
            raise ValueError(
                f"{filename}: missing rows {sorted(expected - actual)}, "
                f"unexpected rows {sorted(actual - expected)}"
            )

    for case, path in sorted(cases.items()):
        try:
            verify_case(
                path,
                {decoder: stats[case, decoder] for decoder in DECODERS},
                {reference: paired[case, reference] for reference in REFERENCES},
                shots=shots,
                seed=seeds[case],
                resample=resample,
            )
        except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
            raise ValueError(f"{case}: {error}") from error
    return len(cases), len(cases) * shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Archive directory containing manifest.json")
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Compare the archive to a seeded replay using the same Stim build and platform",
    )
    args = parser.parse_args()
    try:
        cases, shots = verify_archive(args.archive, resample=args.resample)
    except (OSError, TypeError, ValueError, KeyError, RuntimeError, csv.Error) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(f"Verified {cases} cases with {shots:,} shots")


if __name__ == "__main__":
    main()
