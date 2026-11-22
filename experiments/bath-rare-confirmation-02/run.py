"""Confirm the distance-7 fitted bath rare error rate on fresh Stim shots."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1]
STRESS = EXPERIMENTS / "bath-background-stress-01"
sys.path.insert(0, str(STRESS))
from run import build_channels, detector_counts, replay, wilson_95


P_CHECK = 0.001
SHOTS_PER_LAYOUT = 100_000
CHUNK_SHOTS = 5_000
STIM_SEED = 371_398_201_762_000
ARM = replay.ARMS[1]


def score(circuit, decoder, is_herald, seed, shots):
    detectors, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    predictions = decoder.decode(detectors)
    failed = np.any(predictions[:, : observables.shape[1]] != observables, axis=1)
    return detector_counts(detectors, is_herald, failed)


def run_layout(root, arm, index, layout_seed, draw, bath_ids, control, control_shots):
    upstream = replay.load_upstream(root)
    upstream.P_CHECK = P_CHECK
    true_rates = upstream._rates([arm["channels"][k] for k in draw])
    prior_rates = upstream._rates([arm["priors"][k] for k in draw])
    circuit, is_herald = upstream.build_circuit(
        replay.DISTANCE, replay.BASIS.lower(), *true_rates, upstream.P_MEAS
    )
    prior, is_prior_herald = upstream.build_circuit(
        replay.DISTANCE, replay.BASIS.lower(), *prior_rates, upstream.P_MEAS
    )
    if not np.array_equal(is_herald, is_prior_herald):
        raise ValueError(f"Sampled and prior detector types differ for layout {index + 1}")

    def make_decoder():
        decoder = upstream.ErasureDecoder(prior, is_herald)
        if decoder.skipped:
            raise ValueError(f"Prior for layout {index + 1} has {decoder.skipped} non-graphlike errors")
        return decoder

    control_counts = score(circuit, make_decoder(), is_herald, control["stim_seed"], control_shots)
    if control_counts != {key: control[key] for key in ("failed", "succeeded")}:
        raise ValueError(f"Seeded stress replay differs for layout {index + 1}")
    print(f"Layout {index + 1} control matched ({control_counts['failed']['shots']}/{control_shots})", flush=True)

    decoder = make_decoder()
    counts = {outcome: dict.fromkeys(fields, 0) for outcome, fields in control_counts.items()}
    chunks = []
    for offset in range(0, SHOTS_PER_LAYOUT, CHUNK_SHOTS):
        seed = STIM_SEED + index * 1000 + offset // CHUNK_SHOTS
        chunk = score(circuit, decoder, is_herald, seed, CHUNK_SHOTS)
        for outcome, fields in chunk.items():
            for field, count in fields.items():
                counts[outcome][field] += count
        chunks.append({"stim_seed": seed, "shots": CHUNK_SHOTS, "logical_failures": chunk["failed"]["shots"]})
        if len(chunks) % 5 == 0:
            print(f"Layout {index + 1}: {offset + CHUNK_SHOTS:,}/{SHOTS_PER_LAYOUT:,} fresh shots", flush=True)

    failures = counts["failed"]["shots"]
    return {
        "layout_index": index + 1,
        "layout_seed": layout_seed,
        "bath_ids_in_data_qubit_order": bath_ids,
        "shots": SHOTS_PER_LAYOUT,
        "logical_failures": failures,
        "logical_error_rate": failures / SHOTS_PER_LAYOUT,
        "wilson_95": wilson_95(failures, SHOTS_PER_LAYOUT),
        "detector_counts_by_outcome": counts,
        "original_seed_control": {"stim_seed": control["stim_seed"], **control_counts},
        "chunks": chunks,
    }


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Pinned Sakidja checkout")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        parser.error("--workers must be between 1 and 5")
    if args.output.exists():
        parser.error("Choose a new output file")

    root = args.upstream.resolve()
    revision = git_output(root, "rev-parse", "HEAD")
    if revision != replay.UPSTREAM_COMMIT:
        parser.error(f"Checkout must be at {replay.UPSTREAM_COMMIT}, got {revision}")
    if git_output(root, "status", "--porcelain", "--untracked-files=no"):
        parser.error("Use a checkout with clean tracked files")

    source_bytes = (STRESS / "result.json").read_bytes()
    source = json.loads(source_bytes)
    timing_bytes = (STRESS / "timing" / "result.json").read_bytes()
    timing = json.loads(timing_bytes)
    source_hash = sha256(source_bytes).hexdigest()
    # Git can change line endings when these saved JSON records are checked out.
    source_lf = source_bytes.replace(b"\r\n", b"\n")
    source_hashes = {
        source_hash,
        sha256(source_lf).hexdigest(),
        sha256(source_lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
    if (
        source["upstream_commit"] != revision
        or timing["upstream_commit"] != revision
        or timing["source_sha256"] not in source_hashes
    ):
        raise ValueError("Original stress or timing provenance differs")
    point, = (point for point in source["points"] if (point["arm"], point["p_check"]) == (ARM, P_CHECK))
    timed_point, = (point for point in timing["points"] if (point["arm"], point["p_check"]) == (ARM, P_CHECK))
    if any(timed_point[key] != point[key] for key in ("shots", "logical_failures")):
        raise ValueError("Original timing count differs from the original stress count")

    upstream = replay.load_upstream(root)
    if upstream.P_CHECK != P_CHECK or upstream.P_MEAS != source["p_measurement_flip"]:
        raise ValueError("Native error settings differ from the original stress run")
    versions = {
        "python": platform.python_version(),
        **{name: version(name) for name in ("numpy", "scipy", "stim", "pymatching", "pandas", "qutip")},
    }
    if versions != source["versions"]:
        raise ValueError(f"Use the original environment to replay the seed controls. Found {versions}")

    arm = build_channels(root, upstream, ARM)
    bath_ids = np.asarray(arm["bath_ids"])
    layout_seeds = [upstream.SEED + 1000 * replay.DISTANCE + j for j in range(upstream.N_DRAWS)]
    draws = [np.random.default_rng(seed).integers(0, len(bath_ids), replay.DISTANCE**2) for seed in layout_seeds]
    layouts = [bath_ids[draw].tolist() for draw in draws]
    if layout_seeds != source["layout_seeds"] or layouts != source["layout_bath_ids_in_data_qubit_order"]:
        raise ValueError("Pinned upstream layouts differ from the original stress run")

    results = [None] * len(layout_seeds)
    workers = min(args.workers, len(results))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(run_layout, root, arm, index, seed, draw, layout, control, source["shots_per_layout"]): index
            for index, (seed, draw, layout, control) in enumerate(
                zip(layout_seeds, draws, layouts, point["layout_counts"], strict=True)
            )
        }
        for job in as_completed(jobs):
            row = job.result()
            results[jobs[job]] = row
            print(f"Layout {row['layout_index']} finished: {row['logical_failures']}/{row['shots']}", flush=True)

    shots = sum(row["shots"] for row in results)
    failures = sum(row["logical_failures"] for row in results)
    seeds = [chunk["stim_seed"] for row in results for chunk in row["chunks"]]
    source_seeds = {row["stim_seed"] for record in source["points"] for row in record["layout_counts"]}
    if len(seeds) != len(set(seeds)) or source_seeds.intersection(seeds):
        raise ValueError("Fresh Stim seeds overlap")
    result = {
        "format_version": 1,
        "upstream_repository": source["upstream_repository"],
        "upstream_commit": revision,
        "upstream_source": replay.SOURCE.as_posix(),
        "upstream_table_directory": replay.TABLE_DIR.as_posix(),
        **{
            key: source[key] for key in (
                "attachment", "gamma_sys", "clock_period", "distance", "basis", "syndrome_rounds",
                "p_measurement_flip", "p_reset_flip",
            )
        },
        **{
            key: arm[key] for key in (
                "arm", "mean_check_time", "mean_physical_bath_erasure_probability",
                "max_abs_true_channel_reconstruction_delta",
            )
        },
        "p_check": P_CHECK,
        "shots": shots,
        "logical_failures": failures,
        "logical_error_rate": failures / shots,
        "wilson_95": wilson_95(failures, shots),
        "interval_scope": "Monte Carlo precision conditional on the five fixed layouts",
        "shots_per_layout": SHOTS_PER_LAYOUT,
        "chunk_shots": CHUNK_SHOTS,
        "stim_seed_rule": f"{STIM_SEED} + 1000 * zero-based layout index + zero-based chunk index",
        "parallel_accuracy_workers": workers,
        "layouts": results,
        "environment": {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "versions": versions},
        "original_stress_control": {
            "path": "../bath-background-stress-01/result.json", "sha256": timing["source_sha256"],
            **{key: point[key] for key in ("shots", "logical_failures", "logical_error_rate", "wilson_95")},
            "verification": "Matched logical failures and detector event totals by layout and logical outcome",
        },
        "independent_serial_timing": {
            "path": "../bath-background-stress-01/timing/result.json", "sha256": sha256(timing_bytes).hexdigest(),
            **{key: timed_point[key] for key in ("shots", "logical_failures", "batch_us_shot", "batch_us_round")},
        },
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Confirmation: {failures}/{shots}", flush=True)


if __name__ == "__main__":
    main()
