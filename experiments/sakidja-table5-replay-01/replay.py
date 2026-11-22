"""Fresh Stim shots for two cells of Sakidja's Table 5."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from scipy.stats import beta


UPSTREAM_COMMIT = "94572516042297d6b6018fe76eb2f2b547e3c46f"
TABLE_DIR = Path("part1/results_v3_final8_one_gamma_sys_0.0")
SOURCE = Path("part1/dual_rail_structured_bath.py")
T_CLOCK, DISTANCE, BASIS = 6.0, 7, "Z"
FRESH_SEED = 44448523055381
ARMS = ("Clock, frame calibrated", "Structured bath, ML then fit, balanced check")
PROBABILITIES = ("p_erase", "p_x", "p_y", "p_z")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_upstream(root):
    os.environ["SQB_ATTACH"] = "one"
    spec = importlib.util.spec_from_file_location("sakidja_part1", root / SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interval_95(errors, shots):
    lower = 0.0 if errors == 0 else float(beta.ppf(0.025, errors, shots - errors + 1))
    upper = 1.0 if errors == shots else float(beta.ppf(0.975, errors + 1, shots - errors))
    return [lower, upper]


def replay_arm(root, point_index, table_row, published_seed_mode):
    upstream = load_upstream(root)
    arm = table_row["Arm"]
    data = root / TABLE_DIR
    parameters = {
        int(row["Bath"]): row
        for row in read_rows(data / "raw_parameters_per_bath.csv")
        if row["Set"] == "unseen"
    }
    rows = [
        row for row in read_rows(data / "raw_round_channels_per_bath.csv")
        if float(row["T_clock"]) == T_CLOCK and row["Arm"] == arm
    ]
    rows.sort(key=lambda row: int(row["Bath"]))
    if [int(row["Bath"]) for row in rows] != sorted(parameters):
        raise ValueError(f"Incomplete or duplicated bath rows for {arm}")

    true_channels, prior_channels = [], []
    max_channel_delta = 0.0
    for row in rows:
        bath = parameters[int(row["Bath"])]
        true_params = tuple(float(bath[f"true_{key}"]) for key in upstream.PARAM_NAMES) + (0.0,)
        fit_params = tuple(float(bath[f"fit_{key}"]) for key in upstream.PARAM_NAMES) + (0.0,)
        tau = float(row["Check_time"])
        model_params = true_params if arm == upstream.ARM_CLOCK else fit_params
        frame = upstream.correction(model_params, tau)
        published_channel = {key: float(row[key]) for key in PROBABILITIES}
        recomputed = upstream.round_channel(upstream.liouvillian(true_params), tau, frame)
        expected = dict(published_channel, chi_I=float(row["Unflagged_process_fidelity"]))
        max_channel_delta = max(max_channel_delta, *(abs(recomputed[k] - v) for k, v in expected.items()))
        true_channels.append(published_channel)
        if arm == upstream.ARM_CLOCK:
            prior_channels.append(published_channel)
        else:
            prior_channels.append(
                upstream.round_channel(upstream.liouvillian(fit_params), tau, frame)
            )
    if max_channel_delta > 1e-10:
        raise ValueError(f"Published physical channel differs by {max_channel_delta} for {arm}")

    published_seed = upstream.SEED + point_index
    fresh_seed = FRESH_SEED + point_index
    seed = published_seed if published_seed_mode else fresh_seed
    shots = upstream.SHOTS
    failures = upstream.logical_errors(DISTANCE, BASIS.lower(), true_channels, prior_channels, shots, seed)
    return {
        "arm": arm,
        "mean_check_time": float(np.mean([float(row["Check_time"]) for row in rows])),
        "max_abs_true_channel_reconstruction_delta": max_channel_delta,
        "published_logical_errors": int(table_row["Logical_errors"]),
        "published_stim_seeds": [published_seed + 7919 * j for j in range(upstream.N_DRAWS)],
        "replay_stim_seeds": [seed + 7919 * j for j in range(upstream.N_DRAWS)],
        "replay_logical_errors": failures,
        "replay_logical_error_rate": failures / shots,
        "conditional_binomial_clopper_pearson_95": interval_95(failures, shots),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Local checkout of the Sakidja repository")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    parser.add_argument("--published-seeds", action="store_true", help="Diagnostic rerun with original shot seeds")
    args = parser.parse_args()
    root = args.upstream.resolve()
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if rev != UPSTREAM_COMMIT:
        parser.error(f"Checkout must be at {UPSTREAM_COMMIT}, got {rev}")

    upstream = load_upstream(root)
    table = read_rows(root / TABLE_DIR / "table3_surface_code.csv")
    points = [
        (i, row) for i, row in enumerate(table)
        if float(row["T_clock"]) == T_CLOCK and int(row["Distance"]) == DISTANCE
        and row["Memory_basis"] == BASIS and row["Arm"] in ARMS
    ]
    if tuple(row["Arm"] for _, row in points) != ARMS:
        raise ValueError("Expected both Table 5 entries in upstream sweep order")
    bath_ids = [
        int(row["Bath"]) for row in read_rows(root / TABLE_DIR / "raw_parameters_per_bath.csv")
        if row["Set"] == "unseen"
    ]
    bath_ids = np.array(sorted(bath_ids))
    if len(bath_ids) != upstream.N_TEST:
        raise ValueError("Unexpected count of unseen baths")

    with ProcessPoolExecutor(max_workers=len(points)) as pool:
        work = [pool.submit(replay_arm, root, i, row, args.published_seeds) for i, row in points]
        arms = [job.result() for job in work]

    layout_seeds = [upstream.SEED + 1000 * DISTANCE + j for j in range(upstream.N_DRAWS)]
    result = {
        "upstream_repository": "https://github.com/sakidja/RQL_Quantum_bath",
        "upstream_commit": rev,
        "upstream_source": SOURCE.as_posix(),
        "upstream_table_directory": TABLE_DIR.as_posix(),
        "attachment": upstream.ATTACH,
        "gamma_sys": 0.0,
        "clock_period": T_CLOCK,
        "distance": DISTANCE,
        "basis": BASIS,
        "syndrome_rounds": DISTANCE,
        "shots_per_arm": upstream.SHOTS,
        "shots_per_layout": upstream.SHOTS // upstream.N_DRAWS,
        "stim_seed_mode": "published" if args.published_seeds else "fresh",
        "layout_seeds": layout_seeds,
        "layout_bath_ids_in_data_qubit_order": [
            bath_ids[np.random.default_rng(seed).integers(0, len(bath_ids), DISTANCE**2)].tolist()
            for seed in layout_seeds
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            **{package: version(package) for package in ("stim", "pymatching", "pandas", "qutip")},
        },
        "arms": arms,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for arm in arms:
        print(f'{arm["arm"]}: {arm["replay_logical_errors"]}/{upstream.SHOTS}')


if __name__ == "__main__":
    main()
