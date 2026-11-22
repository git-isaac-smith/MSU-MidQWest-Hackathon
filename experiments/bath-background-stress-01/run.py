"""Stress Sakidja's distance-7 Z memory with a common added check error."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import NormalDist
import subprocess
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sakidja-table5-replay-01"))
import replay


BACKGROUNDS = (0.001, 0.005, 0.01, 0.02, 0.04)
SEED = 94530057721006


def build_channels(root, upstream, arm):
    data = root / replay.TABLE_DIR
    parameters = {
        int(row["Bath"]): row
        for row in replay.read_rows(data / "raw_parameters_per_bath.csv")
        if row["Set"] == "unseen"
    }
    rows = [
        row for row in replay.read_rows(data / "raw_round_channels_per_bath.csv")
        if float(row["T_clock"]) == replay.T_CLOCK and row["Arm"] == arm
    ]
    rows.sort(key=lambda row: int(row["Bath"]))
    bath_ids = [int(row["Bath"]) for row in rows]
    if bath_ids != sorted(parameters) or len(bath_ids) != upstream.N_TEST:
        raise ValueError(f"Unexpected unseen bath rows for {arm}")

    channels, priors, times = [], [], []
    max_delta = 0.0
    for row in rows:
        bath = parameters[int(row["Bath"])]
        true_params = tuple(float(bath[f"true_{key}"]) for key in upstream.PARAM_NAMES) + (0.0,)
        fit_params = tuple(float(bath[f"fit_{key}"]) for key in upstream.PARAM_NAMES) + (0.0,)
        tau = float(row["Check_time"])
        model_params = true_params if arm == upstream.ARM_CLOCK else fit_params
        frame = upstream.correction(model_params, tau)
        published = {key: float(row[key]) for key in replay.PROBABILITIES}
        recomputed = upstream.round_channel(upstream.liouvillian(true_params), tau, frame)
        expected = dict(published, chi_I=float(row["Unflagged_process_fidelity"]))
        max_delta = max(max_delta, *(abs(recomputed[k] - p) for k, p in expected.items()))
        channels.append(published)
        priors.append(
            published if arm == upstream.ARM_CLOCK
            else upstream.round_channel(upstream.liouvillian(fit_params), tau, frame)
        )
        times.append(tau)
    if max_delta > 1e-10:
        raise ValueError(f"Published physical channel differs by {max_delta} for {arm}")
    return {
        "arm": arm,
        "bath_ids": bath_ids,
        "channels": channels,
        "priors": priors,
        "mean_check_time": float(np.mean(times)),
        "mean_physical_bath_erasure_probability": float(np.mean([c["p_erase"] for c in channels])),
        "max_abs_true_channel_reconstruction_delta": max_delta,
    }


def wilson_95(errors, shots):
    z = NormalDist().inv_cdf(0.975)
    proportion = errors / shots
    denominator = 1 + z**2 / shots
    center = (proportion + z**2 / (2 * shots)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / shots + z**2 / (4 * shots**2)) / denominator
    return [max(0.0, float(center - half_width)), min(1.0, float(center + half_width))]


def detector_counts(detectors, is_herald, failed):
    counts = {}
    for label, select in (("failed", failed), ("succeeded", ~failed)):
        subset = detectors[select]
        counts[label] = {
            "shots": len(subset),
            "herald_detector_events": int(np.count_nonzero(subset[:, is_herald])),
            "syndrome_detector_events": int(np.count_nonzero(subset[:, ~is_herald])),
        }
    return counts


def run_point(root, arm_data, background, shots_per_layout, layouts, stim_seeds):
    upstream = replay.load_upstream(root)
    upstream.P_CHECK = background
    layout_counts = []
    for layout, stim_seed in zip(layouts, stim_seeds, strict=True):
        true_rates = upstream._rates([arm_data["channels"][k] for k in layout])
        prior_rates = upstream._rates([arm_data["priors"][k] for k in layout])
        circuit, is_herald = upstream.build_circuit(
            replay.DISTANCE, replay.BASIS.lower(), *true_rates, upstream.P_MEAS
        )
        prior, is_prior_herald = upstream.build_circuit(
            replay.DISTANCE, replay.BASIS.lower(), *prior_rates, upstream.P_MEAS
        )
        if not np.array_equal(is_herald, is_prior_herald):
            raise ValueError("Sampled and prior detector types differ")
        detectors, observables = circuit.compile_detector_sampler(seed=stim_seed).sample(
            shots_per_layout, separate_observables=True
        )
        decoder = upstream.ErasureDecoder(prior, is_herald)
        if decoder.skipped:
            raise ValueError(f"Prior has {decoder.skipped} non-graphlike errors")
        predictions = decoder.decode(detectors)
        failed = np.any(predictions[:, : observables.shape[1]] != observables, axis=1)
        layout_counts.append(
            {"stim_seed": stim_seed, **detector_counts(detectors, is_herald, failed)}
        )

    shots = shots_per_layout * len(layouts)
    failures = sum(layout["failed"]["shots"] for layout in layout_counts)
    detector_means = {}
    for outcome, count in (("failed", failures), ("succeeded", shots - failures)):
        detector_means[outcome] = {
            f"mean_{event}_detector_events_per_shot": (
                sum(layout[outcome][f"{event}_detector_events"] for layout in layout_counts) / count
                if count else None
            )
            for event in ("herald", "syndrome")
        }
    return {
        "arm": arm_data["arm"],
        "p_check": background,
        "shots": shots,
        "logical_failures": failures,
        "logical_error_rate": failures / shots,
        "wilson_95": wilson_95(failures, shots),
        "detector_means_by_outcome": detector_means,
        "layout_counts": layout_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Local checkout of Sakidja's repository")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    parser.add_argument("--shots", type=int, default=20_000, help="Total shots per arm and level")
    parser.add_argument("--workers", type=int, default=10, help="Maximum parallel points")
    args = parser.parse_args()
    root = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != replay.UPSTREAM_COMMIT:
        parser.error(f"Checkout must be at {replay.UPSTREAM_COMMIT}, got {revision}")

    upstream = replay.load_upstream(root)
    if args.shots <= 0 or args.shots % upstream.N_DRAWS:
        parser.error(f"--shots must be a positive multiple of {upstream.N_DRAWS}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    shots_per_layout = args.shots // upstream.N_DRAWS
    arms = [build_channels(root, upstream, arm) for arm in replay.ARMS]
    bath_ids = np.array(arms[0]["bath_ids"])
    layout_seeds = [upstream.SEED + 1000 * replay.DISTANCE + j for j in range(upstream.N_DRAWS)]
    layouts = [
        np.random.default_rng(seed).integers(0, len(bath_ids), replay.DISTANCE**2)
        for seed in layout_seeds
    ]

    points = [None] * (len(arms) * len(BACKGROUNDS))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(points))) as pool:
        work = {}
        for level_index, background in enumerate(BACKGROUNDS):
            for arm_index, arm in enumerate(arms):
                index = level_index * len(arms) + arm_index
                stim_seeds = [
                    SEED + index * upstream.N_DRAWS + j for j in range(upstream.N_DRAWS)
                ]
                work[pool.submit(
                    run_point, root, arm, background, shots_per_layout, layouts, stim_seeds
                )] = index
        for job in as_completed(work):
            point = job.result()
            points[work[job]] = point
            print(
                f'{point["arm"]}, P_CHECK={point["p_check"]}: '
                f'{point["logical_failures"]}/{point["shots"]}',
                flush=True,
            )

    result = {
        "upstream_repository": "https://github.com/sakidja/RQL_Quantum_bath",
        "upstream_commit": revision,
        "upstream_source": replay.SOURCE.as_posix(),
        "upstream_table_directory": replay.TABLE_DIR.as_posix(),
        "attachment": upstream.ATTACH,
        "gamma_sys": 0.0,
        "clock_period": replay.T_CLOCK,
        "distance": replay.DISTANCE,
        "basis": replay.BASIS,
        "syndrome_rounds": replay.DISTANCE,
        "native_p_check": upstream.P_CHECK,
        "p_measurement_flip": upstream.P_MEAS,
        "p_reset_flip": upstream.P_MEAS,
        "shots_per_layout": shots_per_layout,
        "layout_seeds": layout_seeds,
        "layout_bath_ids_in_data_qubit_order": [bath_ids[layout].tolist() for layout in layouts],
        "stim_base_seed": SEED,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "stim", "pymatching", "pandas", "qutip")},
        },
        "arms": [
            {key: arm[key] for key in (
                "arm", "mean_check_time", "mean_physical_bath_erasure_probability",
                "max_abs_true_channel_reconstruction_delta",
            )}
            for arm in arms
        ],
        "points": points,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
