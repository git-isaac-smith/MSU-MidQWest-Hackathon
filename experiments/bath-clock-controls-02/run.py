"""Test whether site-to-time assignment matters in the pinned dual-rail model."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np


HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "sakidja-table5-replay-01" / "replay.py"
SPEC = importlib.util.spec_from_file_location("table5_replay", REPLAY)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

CLOCK, DISTANCE, BASIS = 6.0, 7, "z"
SEED = 92634328730117
INDIVIDUAL = replay.ARMS[1]
CHANNEL_KEYS = (*replay.PROBABILITIES, "chi_I")
ARMS = ("fixed_6", "global", "global_installed_calibration", "shuffled", "individual")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parameters(upstream, row, key):
    return tuple(float(row[f"{key}_{name}"]) for name in upstream.PARAM_NAMES) + (0.0,)


def calibration_costs(upstream, parameters_by_bath, grid):
    result = {}
    for bath, fit in parameters_by_bath.items():
        liouvillian = upstream.liouvillian(fit)
        costs = []
        for tau in grid:
            frame = upstream.correction(fit, tau)
            channel = upstream.round_channel(liouvillian, tau, frame)
            costs.append(sum(channel[key] for key in ("p_x", "p_y", "p_z"))
                         + upstream.ERASURE_COST * channel["p_erase"])
        result[bath] = np.array(costs)
    return result


def selection(costs, grid):
    pooled = np.mean(costs, axis=0)
    best = int(np.argmin(pooled))
    return {"grid_index": best, "time": float(grid[best]), "mean_cost_by_time": pooled.tolist()}


def schedule_calibration(upstream, parameters_path, layout_seeds, shuffle_seeds):
    rows = replay.read_rows(parameters_path)
    train_fits = {int(row["Bath"]): parameters(upstream, row, "fit") for row in rows
                  if row["Set"] == "training" and row["fit_J_sb"]}
    installed_fits = {int(row["Bath"]): parameters(upstream, row, "fit") for row in rows
                      if row["Set"] == "unseen"}
    if sorted(train_fits) != list(range(upstream.N_RAW_FIT)) or len(installed_fits) != upstream.N_TEST:
        raise ValueError("Unexpected saved calibration cohorts")
    if set(train_fits) & set(installed_fits):
        raise ValueError("Training and evaluation bath IDs overlap")
    grid = np.linspace(CLOCK, CLOCK * (1 + upstream.WINDOW_WIDE), upstream.N_TAU_WIDE)
    bath_ids = np.array(sorted(installed_fits))
    layouts = [bath_ids[np.random.default_rng(seed).integers(0, len(bath_ids), DISTANCE**2)].tolist()
               for seed in layout_seeds]

    train_costs = calibration_costs(upstream, train_fits, grid)
    independent_global = selection(list(train_costs.values()), grid)
    print(f"Independent calibration global time: {independent_global['time']}", flush=True)
    installed_costs = calibration_costs(upstream, installed_fits, grid)
    best_by_bath = {bath: float(grid[int(np.argmin(costs))]) for bath, costs in installed_costs.items()}
    installed_global = selection([installed_costs[bath] for layout in layouts for bath in layout], grid)
    print(f"Installed calibration global time: {installed_global['time']}", flush=True)

    frozen = []
    for layout, shuffle_seed in zip(layouts, shuffle_seeds, strict=True):
        individual = [best_by_bath[bath] for bath in layout]
        shuffled = np.random.default_rng(shuffle_seed).permutation(individual).tolist()
        if Counter(individual) != Counter(shuffled) or individual == shuffled:
            raise ValueError("Shuffle must preserve the time multiset and change assignments")
        frozen.append({"bath_ids": layout, "shuffle_seed": shuffle_seed, "times": {
            "fixed_6": [CLOCK] * len(layout),
            "global": [independent_global["time"]] * len(layout),
            "global_installed_calibration": [installed_global["time"]] * len(layout),
            "shuffled": shuffled,
            "individual": individual,
        }})
    return {"time_grid": grid.tolist(), "training_fit_bath_ids": sorted(train_fits),
            "held_out_fit_bath_ids": sorted(installed_fits), "layout_seeds": layout_seeds,
            "cost": {"unflagged_pauli": 1.0, "erasure": upstream.ERASURE_COST},
            "independent_global": independent_global, "installed_global": installed_global,
            "individual_times_by_bath": best_by_bath, "frozen_layouts": frozen}


def physical_and_prior(upstream, arm, tau, true_params, fitted_params, true_L, fitted_L):
    if arm == "fixed_6":
        frame = upstream.correction(true_params, tau)
        physical = upstream.round_channel(true_L, tau, frame)
        return physical, physical
    frame = upstream.correction(fitted_params, tau)
    return upstream.round_channel(true_L, tau, frame), upstream.round_channel(fitted_L, tau, frame)


def channels_for_run(upstream, arm, schedule, saved_parameters, published):
    pairs = {}
    for layout in schedule:
        for bath, tau in zip(layout["bath_ids"], layout["times"][arm], strict=True):
            if (bath, tau) in pairs:
                continue
            row = saved_parameters[bath]
            true = parameters(upstream, row, "true")
            fit = parameters(upstream, row, "fit")
            pairs[bath, tau] = physical_and_prior(
                upstream, arm, tau, true, fit, upstream.liouvillian(true), upstream.liouvillian(fit))
    max_delta = None
    if published:
        max_delta = 0.0
        for bath, row in published.items():
            if (bath, float(row["Check_time"])) not in pairs and any(k == bath for k, _ in pairs):
                raise ValueError(f"Assigned time differs from published time for {arm}, bath {bath}")
            tau = float(row["Check_time"])
            if (bath, tau) in pairs:
                physical = pairs[bath, tau][0]
            else:
                true = parameters(upstream, saved_parameters[bath], "true")
                fit = parameters(upstream, saved_parameters[bath], "fit")
                physical, _ = physical_and_prior(upstream, arm, tau, true, fit,
                                                  upstream.liouvillian(true), upstream.liouvillian(fit))
            saved = {key: float(row[key]) for key in replay.PROBABILITIES}
            saved["chi_I"] = float(row["Unflagged_process_fidelity"])
            max_delta = max(max_delta, *(abs(physical[key] - saved[key]) for key in CHANNEL_KEYS))
        if max_delta > 1e-10:
            raise ValueError(f"Published {arm} channel differs by {max_delta}")
    return pairs, max_delta


def score_layout(upstream, physical, priors, shots, seed):
    circuit, is_herald = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(physical), upstream.P_MEAS)
    prior, prior_is_herald = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(priors), upstream.P_MEAS)
    if not np.array_equal(is_herald, prior_is_herald):
        raise ValueError("Physical and matching-prior circuits have different herald structure")
    decoder = upstream.ErasureDecoder(prior, is_herald)
    if decoder.skipped:
        raise ValueError("Author decoder skipped a non-graphlike error")
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    pred = decoder.decode(dets)
    failures = int(np.sum(np.any(pred[:, :obs.shape[1]] != obs, axis=1)))
    return {"stim_seed": seed, "shots": shots, "logical_errors": failures,
            "physical_circuit_sha256": sha256(str(circuit).encode()).hexdigest(),
            "prior_circuit_sha256": sha256(str(prior).encode()).hexdigest(),
            "sample_sha256": sha256(np.packbits(dets).tobytes() + np.packbits(obs).tobytes()).hexdigest()}


def describe_times(times):
    return {"min": min(times), "mean": float(np.mean(times)), "max": max(times),
            "quantiles_25_50_75": np.quantile(times, [0.25, 0.5, 0.75]).tolist(),
            "count_by_time": {f"{tau:.2f}": count for tau, count in sorted(Counter(times).items())}}


def run_arm(upstream, arm, calibration, pairs, shots, stim_seeds):
    rows = []
    for layout, seed in zip(calibration["frozen_layouts"], stim_seeds, strict=True):
        times = layout["times"][arm]
        channels = [pairs[bath, tau] for bath, tau in zip(layout["bath_ids"], times, strict=True)]
        physical, priors = zip(*channels, strict=True)
        result = score_layout(upstream, physical, priors, shots, seed)
        result.update(time_distribution=describe_times(times),
                      mean_physical_channel={key: float(np.mean([c[key] for c in physical])) for key in CHANNEL_KEYS},
                      mean_prior_channel={key: float(np.mean([c[key] for c in priors])) for key in CHANNEL_KEYS})
        rows.append(result)
        print(f"{arm} layout {len(rows)}/{upstream.N_DRAWS}: {result['logical_errors']}/{shots}", flush=True)
    failures = sum(row["logical_errors"] for row in rows)
    pooled_shots = shots * len(rows)
    return {"arm": arm, "shots": pooled_shots, "logical_errors": failures, "ler": failures / pooled_shots,
            "conditional_binomial_clopper_pearson_95": replay.interval_95(failures, pooled_shots),
            "time_distribution": describe_times([tau for layout in calibration["frozen_layouts"]
                                                  for tau in layout["times"][arm]]),
            "mean_physical_channel": {key: float(np.mean([row["mean_physical_channel"][key] for row in rows]))
                                      for key in CHANNEL_KEYS},
            "layouts": rows}


def load_pinned(root):
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if rev != replay.UPSTREAM_COMMIT:
        raise ValueError(f"Expected pinned upstream commit {replay.UPSTREAM_COMMIT}, got {rev}")
    tracked = [replay.SOURCE, replay.TABLE_DIR / "raw_parameters_per_bath.csv",
               replay.TABLE_DIR / "raw_round_channels_per_bath.csv"]
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *map(str, tracked)], check=True)
    upstream = replay.load_upstream(root)
    if (upstream.ATTACH, upstream.GAMMA_SYS_LIST, upstream.P_CHECK, upstream.P_MEAS) != ("one", (0.0,), 0.001, 0.001):
        raise ValueError("Unexpected channel configuration")
    if (upstream.SHOTS, upstream.N_DRAWS) != (20_000, 5):
        raise ValueError("Unexpected shot configuration")
    return upstream, rev, root / tracked[1], root / tracked[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    upstream, rev, params_path, channels_path = load_pinned(args.upstream.resolve())

    seed_children = np.random.SeedSequence(SEED).spawn(1 + len(ARMS))
    shuffle_seeds = seed_children[0].generate_state(upstream.N_DRAWS, dtype=np.uint64).tolist()
    seeds = {arm: child.generate_state(upstream.N_DRAWS, dtype=np.uint64).tolist()
             for arm, child in zip(ARMS, seed_children[1:], strict=True)}
    all_seeds = [*shuffle_seeds, *(seed for arm_seeds in seeds.values() for seed in arm_seeds)]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Random seeds overlap")
    layout_seeds = [upstream.SEED + 1000 * DISTANCE + j for j in range(upstream.N_DRAWS)]
    args.out.mkdir(exist_ok=True, parents=True)
    calibration = schedule_calibration(upstream, params_path, layout_seeds, shuffle_seeds)
    calibration.update(upstream_commit=rev, root_seed=SEED, stim_seeds=seeds,
                       parameter_csv_sha256=sha256(params_path.read_bytes()).hexdigest())
    calibration_path = args.out / "calibration.json"
    write_json(calibration_path, calibration)

    saved_parameters = {int(row["Bath"]): row for row in replay.read_rows(params_path) if row["Set"] == "unseen"}
    published_rows = [row for row in replay.read_rows(channels_path) if float(row["T_clock"]) == CLOCK]
    published_by_arm = {arm: {int(row["Bath"]): row for row in published_rows if row["Arm"] == label}
                        for arm, label in (("fixed_6", upstream.ARM_CLOCK), ("individual", INDIVIDUAL))}
    for arm, rows in published_by_arm.items():
        if set(rows) != set(saved_parameters):
            raise ValueError(f"Missing published channels for {arm}")
    for bath, tau in calibration["individual_times_by_bath"].items():
        if tau != float(published_by_arm["individual"][bath]["Check_time"]):
            raise ValueError(f"Reconstructed fitted check time differs for bath {bath}")

    result = {"upstream_commit": rev, "calibration_sha256": sha256(calibration_path.read_bytes()).hexdigest(),
              "versions": {"python": platform.python_version(), **{package: version(package) for package in
                           ("numpy", "scipy", "pandas", "qutip", "stim", "pymatching")}},
              "attachment": upstream.ATTACH, "gamma_sys": 0.0, "distance": DISTANCE, "basis": BASIS,
              "rounds": DISTANCE, "p_check": upstream.P_CHECK, "p_meas": upstream.P_MEAS,
              "shots_per_layout": upstream.SHOTS // upstream.N_DRAWS, "arms": []}
    for arm in ARMS:
        pairs, delta = channels_for_run(upstream, arm, calibration["frozen_layouts"], saved_parameters,
                                        published_by_arm.get(arm))
        measured = run_arm(upstream, arm, calibration, pairs, result["shots_per_layout"], seeds[arm])
        measured["max_abs_published_channel_reconstruction_delta"] = delta
        result["arms"].append(measured)
    fingerprints = [row["sample_sha256"] for arm in result["arms"] for row in arm["layouts"]]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Evaluation sample fingerprints overlap")
    write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
