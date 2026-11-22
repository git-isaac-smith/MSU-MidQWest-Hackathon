"""Time the original bath-stress inference calls on exactly replayed records."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns


# Keep both the benchmark and the parent runner import local to this process.
sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from run import build_channels, detector_counts, replay


SOURCE = Path(__file__).resolve().parents[1] / "result.json"
WARMUP_SHOTS = 64


def time_layout(upstream, source, point, layout, arm, saved):
    bath_index = {bath_id: index for index, bath_id in enumerate(arm["bath_ids"])}
    channels = [arm["channels"][bath_index[bath_id]] for bath_id in layout]
    priors = [arm["priors"][bath_index[bath_id]] for bath_id in layout]
    distance, basis = source["distance"], source["basis"].lower()
    circuit, is_herald = upstream.build_circuit(
        distance, basis, *upstream._rates(channels), upstream.P_MEAS
    )
    prior, is_prior_herald = upstream.build_circuit(
        distance, basis, *upstream._rates(priors), upstream.P_MEAS
    )
    if not np.array_equal(is_herald, is_prior_herald):
        raise ValueError("Sampled and prior detector types differ")

    seed = saved["stim_seed"]
    warmup_seed = seed + (1 << 48)
    shots = source["shots_per_layout"]
    detectors, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    warmup = circuit.compile_detector_sampler(seed=warmup_seed).sample(WARMUP_SHOTS)
    decoder = upstream.ErasureDecoder(prior, is_herald)
    if decoder.skipped:
        raise ValueError(f"Prior has {decoder.skipped} non-graphlike errors")
    decoder.decode(warmup)

    start = perf_counter_ns()
    predictions = decoder.decode(detectors)
    elapsed_ns = perf_counter_ns() - start

    failed = np.any(predictions[:, : observables.shape[1]] != observables, axis=1)
    counts = detector_counts(detectors, is_herald, failed)
    if counts != {outcome: saved[outcome] for outcome in ("failed", "succeeded")}:
        raise ValueError(f"Replay differs for {point['arm']}, P_CHECK={point['p_check']}, seed {seed}")
    return {
        "stim_seed": seed,
        "warmup_stim_seed": warmup_seed,
        "detector_shape": list(detectors.shape),
        "detector_dtype": str(detectors.dtype),
        "herald_detectors": int(np.count_nonzero(is_herald)),
        "observable_shape": list(observables.shape),
        "logical_failures": counts["failed"]["shots"],
        "shots": shots,
        "decode_ns": elapsed_ns,
        "batch_us_shot": elapsed_ns / (1000 * shots),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Pinned upstream checkout")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    root = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    if revision != source["upstream_commit"] or revision != replay.UPSTREAM_COMMIT:
        parser.error(f"Upstream checkout must be at {source['upstream_commit']}")
    versions = {
        "python": platform.python_version(),
        **{name: version(name) for name in ("numpy", "scipy", "stim", "pymatching", "pandas", "qutip")},
    }
    if versions != source["versions"]:
        parser.error(f"Use the source package versions for the seeded replay. Found {versions}")

    upstream = replay.load_upstream(root)
    arms = {arm: build_channels(root, upstream, arm) for arm in replay.ARMS}
    points = []
    for point in source["points"]:
        upstream.P_CHECK = point["p_check"]
        timings = [
            dict(
                time_layout(upstream, source, point, layout, arms[point["arm"]], saved),
                layout_seed=layout_seed,
            )
            for layout, layout_seed, saved in zip(
                source["layout_bath_ids_in_data_qubit_order"],
                source["layout_seeds"],
                point["layout_counts"],
                strict=True,
            )
        ]
        failures = sum(row["logical_failures"] for row in timings)
        shots = sum(row["shots"] for row in timings)
        if (failures, shots) != (point["logical_failures"], point["shots"]):
            raise ValueError("Cell total differs from the saved result")
        elapsed_ns = sum(row["decode_ns"] for row in timings)
        us_shot = elapsed_ns / (1000 * shots)
        points.append({
            "arm": point["arm"],
            "p_check": point["p_check"],
            "shots": shots,
            "logical_failures": failures,
            "decode_ns": elapsed_ns,
            "batch_us_shot": us_shot,
            "batch_us_round": us_shot / source["syndrome_rounds"],
            "layout_timings": timings,
        })
        print(f'{point["arm"]}, P_CHECK={point["p_check"]}: {failures}/{shots}, {us_shot:.2f} µs/shot', flush=True)

    result = {
        "format_version": 1,
        "source": "../result.json",
        "source_sha256": sha256(source_bytes).hexdigest(),
        "upstream_commit": revision,
        "distance": source["distance"],
        "syndrome_rounds": source["syndrome_rounds"],
        "basis": source["basis"],
        "layout_bath_ids_in_data_qubit_order": source["layout_bath_ids_in_data_qubit_order"],
        "protocol": {
            "timed_operation": "ErasureDecoder.decode(detectors)",
            "timer": "perf_counter_ns wall time",
            "parallel_workers": 1,
            "repeats_per_layout": 1,
            "shots_per_layout_call": source["shots_per_layout"],
            "warmup_shots_per_layout": WARMUP_SHOTS,
            "aggregation": "Sum five timed layout calls and divide by total shots",
            "excluded": ["circuit construction", "sampling", "decoder construction", "warmup", "scoring"],
            "verification": "Failure counts and detector event totals by outcome match the original per layout and point",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "versions": versions,
        },
        "points": points,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
