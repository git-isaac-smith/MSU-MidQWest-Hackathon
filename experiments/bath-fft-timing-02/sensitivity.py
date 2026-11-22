"""Finite calibration shots for the same phase-preserving Fourier timing rule."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import run

BUDGETS = (1000, 10000)
REPEATS = 16
SEED = 2566112202


def draw(signal, shots, rng):
    # Balanced-input logical Pauli measurement gives +1, -1, or 0 (erasure).
    # The R-preparation occupancy measurement is Bernoulli on its own shots.
    occupancy = np.clip(signal[2], 0, 1)
    in_rails = (1 + occupancy) / 2
    retained = rng.binomial(shots, np.broadcast_to(in_rails, signal[:2].shape))
    plus_fraction = np.clip((1 + signal[:2] / in_rails) / 2, 0, 1)
    plus = rng.binomial(retained, plus_fraction)
    measured = np.empty_like(signal)
    measured[:2] = (2 * plus - retained) / shots
    measured[2] = rng.binomial(shots, occupancy) / shots
    return measured


def true_grid(m, params):
    # Separate forward calculation in the no-jump subspace. Jumps go to vacuum.
    h, cs = m.sector_model(params)
    effective = h - 0.5j * sum(c.conj().T @ c for c in cs)
    propagator = expm(-1j * effective * run.GRID[0])
    step = expm(-1j * effective * (run.GRID[1] - run.GRID[0]))
    amplitude = []
    for _ in run.GRID:
        amplitude.append(propagator[m.I_R, m.I_R] * propagator[m.I_L, m.I_L].conjugate())
        propagator = step @ propagator
    return np.array(amplitude)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("upstream", type=Path)
    cli.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = run.model(args.upstream)
    data = args.upstream / run.TABLE
    basis = run.spectrum_basis()
    predictions = []
    for item in run.read(data / "raw_parameters_per_bath.csv"):
        if item["Set"] != "unseen":
            continue
        bath = int(item["Bath"])
        params = tuple(float(item["true_" + n]) for n in m.PARAM_NAMES) + (0.0,)
        s = m.calibration_signal(params)
        signal = np.stack((s[0], s[1], -s[8]))
        amplitude = true_grid(m, params)
        erasure = (1 - abs(amplitude)**2) / 2
        for shots in BUDGETS:
            for rep in range(REPEATS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, bath, shots, rep]))
                measured = draw(signal, shots, rng)
                result, _ = run.infer(measured, basis)
                t = result["fft_time"]
                i = np.flatnonzero(run.GRID == t)[0]
                rotated = amplitude[i] * np.exp(-1j * result["fft_relative_phase_rad"])
                conditional_z = abs(1 - rotated)**2 / (2 * (1 + abs(rotated)**2))
                predictions.append(dict(bath=bath, shots=shots, replicate=rep, fft_time=t,
                                        clipped_candidates=result["clipped_candidates"],
                                        true_erasure=erasure[i], excess_erasure=erasure[i]-min(erasure),
                                        true_conditional_z_with_estimated_frame=conditional_z))

    reference = {int(r["Bath"]):float(r["Check_time"]) for r in run.read(data / "raw_round_channels_per_bath.csv")
                 if float(r["T_clock"]) == 6 and r["Arm"] == run.FIT}
    for r in predictions:
        r["fit_time"] = reference[r["bath"]]
        r["time_error"] = abs(r["fft_time"] - r["fit_time"])
    with (args.out / "sensitivity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0])
        writer.writeheader()
        writer.writerows(predictions)
    summary = dict(budgets=BUDGETS, replicates_per_bath=REPEATS, bath_count=49, root_seed=SEED,
                   seeds="SeedSequence([root_seed, bath_id, shots_per_time_per_observable, replicate])",
                   grid_step=0.06, results={})
    for shots in BUDGETS:
        rows = [r for r in predictions if r["shots"] == shots]
        delta = np.array([r["time_error"] for r in rows])
        summary["results"][shots] = dict(records=len(rows), exact_grid=int(sum(delta < 1e-8)),
            within_one_step=int(sum(delta <= 0.06000001)), mae=float(np.mean(delta)),
            median_abs_error=float(np.median(delta)), max_abs_error=float(max(delta)),
            mean_true_erasure=np.mean([r["true_erasure"] for r in rows]),
            mean_excess_erasure=np.mean([r["excess_erasure"] for r in rows]),
            mean_true_conditional_z_with_estimated_frame=np.mean([r["true_conditional_z_with_estimated_frame"] for r in rows]),
            records_with_clipping=sum(r["clipped_candidates"] > 0 for r in rows))
    (args.out / "sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
