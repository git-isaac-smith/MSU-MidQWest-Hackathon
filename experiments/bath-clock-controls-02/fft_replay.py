"""Test FFT and dominant-tone schedules with the controls' fitted frames and priors."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from run import (BASIS, DISTANCE, HERE, channels_for_run, load_pinned, parameters,
                 replay, run_arm, write_json)


ARMS = {"fft_timing": "fft_time", "dominant_tone_timing": "tone_time", "fft_only": "fft_time"}
SEED = 878641291743


def fft_channels(upstream, schedule, fft_inputs, bath_params):
    pairs, checks = {}, []
    for bath, tau in schedule.items():
        predicted = fft_inputs[bath]
        phase, erasure, conditional_z = (predicted[key] for key in ("phase", "p_erase", "p_z"))
        frame = np.diag([1, np.exp(1j * phase)])
        if not (np.isfinite(phase) and 0 <= erasure <= 1 and 0 <= conditional_z <= 1):
            raise ValueError(f"Invalid FFT prior for bath {bath}")
        prior = dict(p_erase=erasure, p_z=conditional_z, p_x=0.0, p_y=0.0,
                     chi_I=(1 - erasure) * (1 - conditional_z))
        true = parameters(upstream, bath_params[bath], "true")
        exact_frame = upstream.correction(true, tau)
        exact_phase = np.angle(exact_frame[1, 1] * exact_frame[0, 0].conj())
        physical = upstream.round_channel(upstream.liouvillian(true), tau, frame)
        phase_error = float(abs(np.angle(np.exp(1j * (phase - exact_phase)))))
        checks.append(dict(bath=bath, relative_frame_phase_error_rad=phase_error,
                           abs_prior_physical_delta={key: abs(prior[key] - physical[key]) for key in prior}))
        pairs[bath, tau] = physical, prior
    if max(row["relative_frame_phase_error_rad"] for row in checks) > 1e-5:
        raise ValueError("FFT correction phase disagrees with the direct source convention")
    return pairs, checks


def graph_comparison(upstream, layouts, fft_pairs, fitted_pairs):
    comparisons = []
    for layout in layouts:
        sites = list(zip(layout["bath_ids"], layout["times"]["fft_only"], strict=True))
        graphs = []
        for pairs in (fft_pairs, fitted_pairs):
            priors = [pairs[site][1] for site in sites]
            circuit, flags = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(priors), upstream.P_MEAS)
            graphs.append(upstream.ErasureDecoder(circuit, flags))
        fft, fitted = graphs
        shared_edges = fft.edges.keys() & fitted.edges.keys()
        comparisons.append({"same_edges_and_fault_ids": fft.edges.keys() == fitted.edges.keys()
                            and all(fft.edges[e][1] == fitted.edges[e][1] for e in shared_edges),
                            "same_herald_conditioning": fft.cond == fitted.cond,
                            "max_abs_common_edge_weight_difference": max(
                                abs(fft.edges[e][0] - fitted.edges[e][0]) for e in shared_edges)})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("per_bath", type=Path, help="Frozen per-bath FFT timing predictions")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    upstream, rev, params_path, _ = load_pinned(args.upstream.resolve())
    calibration_path = args.out / "calibration.json"
    calibration_bytes = calibration_path.read_bytes()
    calibration = json.loads(calibration_bytes)
    if calibration["upstream_commit"] != rev or calibration["parameter_csv_sha256"] != sha256(params_path.read_bytes()).hexdigest():
        raise ValueError("Frozen controls use different upstream inputs")

    rows = replay.read_rows(args.per_bath)
    fft_inputs = {int(row["bath"]): dict(phase=float(row["fft_relative_phase_rad"]),
                                        p_erase=float(row["fft_erasure"]), p_z=float(row["fft_conditional_z"]))
                  for row in rows}
    grid = np.array(calibration["time_grid"])
    schedules = {}
    for arm, column in ARMS.items():
        times = {int(row["bath"]): float(row[column]) for row in rows}
        if len(times) != len(rows) or sorted(times) != calibration["held_out_fit_bath_ids"]:
            raise ValueError("Expected exactly one timing prediction per evaluation bath")
        if any(not np.isfinite(tau) or np.min(np.abs(grid - tau)) > 1e-10 for tau in times.values()):
            raise ValueError(f"{arm} check time outside the prescribed grid")
        for layout in calibration["frozen_layouts"]:
            layout["times"][arm] = [times[bath] for bath in layout["bath_ids"]]
        schedules[arm] = times

    seed_children = np.random.SeedSequence(SEED).spawn(len(ARMS))
    seeds = {arm: child.generate_state(upstream.N_DRAWS, dtype=np.uint64).tolist()
             for arm, child in zip(ARMS, seed_children, strict=True)}
    timing_seeds = [seed for arm in seeds.values() for seed in arm]
    control_seeds = {seed for arm in calibration["stim_seeds"].values() for seed in arm}
    if set(timing_seeds) & control_seeds or len(set(timing_seeds)) != len(timing_seeds):
        raise ValueError("FFT evaluation seeds overlap")
    frozen = {"upstream_commit": rev, "control_calibration_sha256": sha256(calibration_bytes).hexdigest(),
              "fft_predictions_sha256": sha256(args.per_bath.read_bytes()).hexdigest(),
              "times_by_arm_and_bath": schedules, "fft_frame_and_prior_by_bath": fft_inputs,
              "root_seed": SEED, "stim_seeds": seeds}
    frozen_path = args.out / "fft_calibration.json"
    write_json(frozen_path, frozen)

    saved_parameters = {int(row["Bath"]): row for row in replay.read_rows(params_path) if row["Set"] == "unseen"}
    result = {"fft_calibration_sha256": sha256(frozen_path.read_bytes()).hexdigest(), "arms": []}
    layouts = calibration["frozen_layouts"]
    fitted_fft_pairs, _ = channels_for_run(upstream, "fft_timing", layouts, saved_parameters, None)
    for arm in ARMS:
        if arm == "fft_only":
            pairs, checks = fft_channels(upstream, schedules[arm], fft_inputs, saved_parameters)
            result.update(fft_frame_and_prior_crosscheck=checks,
                          fft_vs_fitted_decoder_graph=graph_comparison(upstream, layouts, pairs, fitted_fft_pairs))
        elif arm == "fft_timing":
            pairs = fitted_fft_pairs
        else:
            pairs, _ = channels_for_run(upstream, arm, layouts, saved_parameters, None)
        measured = run_arm(upstream, arm, calibration, pairs, upstream.SHOTS // upstream.N_DRAWS, seeds[arm])
        measured["evaluation_sites_matching_individual_time"] = sum(
            a == b for layout in calibration["frozen_layouts"]
            for a, b in zip(layout["times"][arm], layout["times"]["individual"], strict=True))
        result["arms"].append(measured)
    write_json(args.out / "fft_result.json", result)


if __name__ == "__main__":
    main()
