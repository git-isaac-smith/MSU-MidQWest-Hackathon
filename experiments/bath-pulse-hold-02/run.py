"""Common pi pulses after a successful erasure check, in the published cold-bath model."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import scipy
from scipy.linalg import expm, polar


REV = "94572516042297d6b6018fe76eb2f2b547e3c46f"
TABLE = "part1/results_v3_final8_one_gamma_sys_0.0"
PULSES = (0, 2, 8, 32, 128)
STEPS = 256


def generator(model, params):
    h, jumps = model.sector_model(params)
    # In this cold model all jumps end in vacuum, which cannot feed back.
    if any(np.any(c[1:, :]) or np.any(c[:, 0]) for c in jumps) or np.any(h[1:, 0]):
        raise ValueError("No-jump propagation requires an absorbing vacuum")
    return -1j * h - .5 * sum(c.conj().T @ c for c in jumps)


def propagate(step, state, pulses, catch_mass):
    state = state.copy()
    survival = [np.vdot(state[1:3], state[1:3]).real / catch_mass]
    for i in range(1, STEPS + 1):
        state = step @ state
        if pulses and i % (STEPS // pulses) == 0:
            state[1:3] *= -1
        survival.append(np.vdot(state[1:3], state[1:3]).real / catch_mass)
    return state, np.array(survival)


def rail_state(block, dim):
    state = np.zeros((dim, 2), dtype=complex)
    state[1:3] = block
    return state


def liouville_check(model, params, tau, launch, n, predicted):
    """Independently evolve an encoded maximally mixed catch through the full master equation."""
    rho = np.zeros((model.DIM, model.DIM), dtype=complex)
    rho[1:3, 1:3] = launch @ launch.conj().T / np.vdot(launch, launch).real
    pulse = np.ones(model.DIM)
    pulse[1:3] = -1
    step = expm(model.liouvillian(params) * tau / max(1, n))
    for _ in range(max(1, n)):
        rho = (step @ rho.ravel()).reshape(rho.shape)
        if n:
            rho *= np.outer(pulse, pulse)
    if abs(np.trace(rho[1:3, 1:3]).real - predicted) > 1e-9 or abs(rho.trace()-1) > 1e-9:
        raise ValueError("Full Lindblad equation disagrees with rail survival")


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() != REV:
        parser.error("Wrong upstream revision")
    os.environ["SQB_ATTACH"] = "one"
    spec = importlib.util.spec_from_file_location("pinned_bath", args.upstream / "part1/dual_rail_structured_bath.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    source = args.upstream / TABLE
    params = {int(r["Bath"]): r for r in read(source / "raw_parameters_per_bath.csv") if r["Set"] == "unseen"}
    rows = [r for r in read(source / "raw_round_channels_per_bath.csv")
            if float(r["T_clock"]) == 6 and r["Arm"] == "Structured bath, ML then fit, balanced check"]
    if sorted(int(r["Bath"]) for r in rows) != sorted(params) or len(params) != 49:
        raise ValueError("Expected the 49 held-out baths")
    records, release = [], []
    for i, row in enumerate(rows):
        idx, tau = int(row["Bath"]), float(row["Check_time"])
        p, fitted = [tuple(float(params[idx][f"{prefix}_{k}"]) for k in model.PARAM_NAMES) + (0.,)
                     for prefix in ("true", "fit")]
        true_g, fit_g = generator(model, p), generator(model, fitted)
        frame = model.correction(fitted, tau)
        launch, fit_launch = [frame.conj().T @ expm(g * tau)[1:3, 1:3] for g in (true_g, fit_g)]
        catch_mass = np.vdot(launch, launch).real / 2
        if abs(1 - catch_mass - float(row["p_erase"])) > 1e-9:
            raise ValueError("Catch probability disagrees with the published channel")
        step, fit_step = [expm(g * tau / STEPS) for g in (true_g, fit_g)]
        for n in PULSES:
            state, survival = propagate(step, rail_state(launch, model.DIM), n, 2 * catch_mass)
            final = state[1:3]
            calibrated, _ = propagate(fit_step, rail_state(fit_launch, model.DIM), n, np.vdot(fit_launch, fit_launch).real)
            frame_end = polar(calibrated[1:3])[0]
            chi_pass = abs(np.trace(frame_end.conj().T @ final))**2 / (2*np.vdot(final, final).real)
            if i == 0:
                liouville_check(model, p, tau, launch, n, survival[-1])
            records.append(dict(bath=idx, check_time=tau, pulses=n, catch_probability=catch_mass,
                                final_loss_given_catch=1-survival[-1], peak_loss_given_catch=1-min(survival),
                                conditional_entanglement_infidelity=1-chi_pass))
            if n == 128:
                for next_n in (0, 128):
                    _, second = propagate(step, state, next_n, 2 * catch_mass)
                    release.append(dict(bath=idx, second_interval_pulses=next_n,
                                        second_interval_peak_loss=1-min(second), second_interval_end_loss=1-second[-1]))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "per-bath.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (args.out / "release.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(release[0]))
        writer.writeheader()
        writer.writerows(release)
    summary = []
    for n in PULSES:
        group = [r for r in records if r["pulses"] == n]
        summary.append(dict(pulses=n, **{name: np.mean([r[name] for r in group]) for name in
                           ("final_loss_given_catch", "peak_loss_given_catch", "conditional_entanglement_infidelity")}))
    (args.out / "summary.json").write_text(json.dumps(dict(upstream_commit=REV, baths=len(rows),
        hold="one additional fitted return interval per bath", grid_intervals=STEPS, numpy=np.__version__,
        scipy=scipy.__version__, weighting="equal bath weights, probabilities conditional on initial pass",
        initial_state="maximally mixed encoded input or normalized Choi for entanglement fidelity",
        check="full Lindblad comparison on first bath at every pulse count; all catch rates match upstream",
        release_after_128=[dict(second_interval_pulses=n, mean_peak_loss=np.mean(
            [r["second_interval_peak_loss"] for r in release if r["second_interval_pulses"] == n])) for n in (0, 128)],
        summary=summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
