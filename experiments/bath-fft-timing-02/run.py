"""Reconstruct balanced-check timing from noiseless rail calibration records."""
import argparse
import csv
import importlib.util
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PIN = "94572516042297d6b6018fe76eb2f2b547e3c46f"
TABLE = "part1/results_v3_final8_one_gamma_sys_0.0"
FIT = "Structured bath, ML then fit, balanced check"
EXACT = "Structured bath, exact, balanced check (bound)"
GRID = np.linspace(6, 15, 151)
TIME = np.linspace(0, 1500, 5000)
DT = TIME[1]
EXT_LEN = 2 * (len(TIME) - 1)


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def model(root):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected {PIN}, got {revision}")
    os.environ.update(SQB_ATTACH="one", SQB_INPUTS="three_inputs")
    spec = importlib.util.spec_from_file_location("sakidja_part1", root / "part1/dual_rail_structured_bath.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spectrum_basis():
    frequencies = np.fft.rfftfreq(EXT_LEN, DT)
    basis = np.exp(2j * np.pi * frequencies[:, None] * GRID) / EXT_LEN
    basis[1:-1] *= 2
    return basis


def infer(signal, basis):
    # Extend X and occupancy evenly and Y oddly at preparation. A far-end taper
    # starts at t=750, well beyond the 6..15 decision window, and joins at zero.
    taper = np.ones_like(TIME)
    late = TIME > 750
    taper[late] = (1 + np.cos(np.pi * (TIME[late] - 750) / 750)) / 2
    tapered = signal * taper
    parity = np.array([[1], [-1], [1]])
    ext = np.concatenate((tapered, parity * tapered[:, -2:0:-1]), axis=1)
    coefficients = np.fft.rfft(ext, axis=1)
    x, y, occupation = (coefficients @ basis).real
    raw_magnitude = np.hypot(x, y)
    magnitude = np.minimum(raw_magnitude, 1.0)
    erasure = (1 - magnitude**2) / 2
    conditional_z = (1 - magnitude)**2 / (2 * (1 + magnitude**2))
    cost = conditional_z + erasure / 8
    i = int(np.argmin(cost))

    # Naive baseline: strongest positive-frequency line in the measured R return.
    raw = np.fft.rfft(signal[2] - signal[2].mean()) / len(TIME)
    freq = np.fft.rfftfreq(len(TIME), DT)
    k = 1 + int(np.argmax(abs(raw[1:])))
    tone = signal[2].mean() + 2 * (raw[k] * np.exp(2j * np.pi * freq[k] * GRID)).real
    row = dict(fft_time=GRID[i], tone_time=GRID[np.argmax(tone)], tone_frequency=freq[k],
               tone_phase_rad=np.angle(raw[k]), tone_amplitude=2 * abs(raw[k]),
               clipped_candidates=int(np.count_nonzero(raw_magnitude > 1)), fft_return_magnitude=magnitude[i], fft_relative_phase_rad=np.arctan2(y[i], x[i]),
               fft_erasure=erasure[i], fft_conditional_z=conditional_z[i], fft_cost=cost[i],
               max_occupancy_identity_error=np.max(abs(occupation - magnitude**2)))
    return row, (occupation, tone, freq, raw)


def stats(rows, method):
    delta = np.array([r[method] - r["fit_time"] for r in rows])
    return dict(exact_grid=int(np.sum(abs(delta) < 1e-8)),
                within_one_step=int(np.sum(abs(delta) <= 0.06 + 1e-8)),
                mae=float(np.mean(abs(delta))), max_error=float(np.max(abs(delta))),
                time_range=[min(r[method] for r in rows), max(r[method] for r in rows)])


def figure(out, example, rows):
    bath, signal, (occ, tone, freq, raw) = example
    row = rows[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.7), layout="constrained")
    ax = axes[0]
    use = (TIME >= 5.7) & (TIME <= 15.3)
    ax.plot(TIME[use], signal[2, use], ".", color="#086878", label="Simulated samples")
    ax.plot(GRID, occ, color="#086878", label="Full FFT")
    ax.plot(GRID, tone, "--", color="#a95428", label="Single tone")
    ax.axvline(row["fit_time"], color="#3c4057", ls=":", label="ML + fit time")
    ax.set(title=f"Rail-R return · bath {bath}", xlabel="Time (model units)", ylabel="Occupation probability")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    ax.plot(freq[1:], 2 * abs(raw[1:]), color="#086878", lw=1)
    ax.scatter(row["tone_frequency"], row["tone_amplitude"], color="#a95428", zorder=3)
    ax.set(title="Rail-R occupation spectrum", xlabel="Frequency (cycles / model time)", ylabel="Amplitude (DC removed)")
    ax = axes[2]
    ref = [r["fit_time"] for r in rows]
    ax.plot([6, 15], [6, 15], color="#3c4057", ls=":", lw=1)
    ax.scatter(ref, [r["tone_time"] for r in rows], color="#a95428", s=18, marker="x", label="Single tone")
    ax.scatter(ref, [r["fft_time"] for r in rows], s=24, facecolors="none", edgecolors="#086878", label="Full FFT")
    ax.set(title="49 unseen baths", xlabel="ML + fit time", ylabel="Signal prediction", aspect="equal")
    ax.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "timing.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    m = model(args.upstream)
    data = args.upstream / TABLE
    args.out.mkdir(exist_ok=True, parents=True)
    basis = spectrum_basis()
    rows = []
    example = None
    for p in read(data / "raw_parameters_per_bath.csv"):
        if p["Set"] != "unseen":
            continue
        bath = int(p["Bath"])
        params = tuple(float(p["true_" + n]) for n in m.PARAM_NAMES) + (0.0,)
        s = m.calibration_signal(params)
        signal = np.stack((s[0], s[1], -s[8]))  # plus X,Y; rail-R-prepared occupation
        row, details = infer(signal, basis)
        # Direct physical-channel cross-check, only after signal-only selection.
        tau = row["fft_time"]
        frame = m.correction(params, tau)
        relative_phase = np.angle(frame[1, 1] * frame[0, 0].conjugate())
        row["frame_phase_error_rad"] = abs(np.angle(np.exp(1j * (row["fft_relative_phase_rad"] - relative_phase))))
        ch = m.round_channel(m.liouvillian(params), tau, frame)
        row["direct_erasure"] = ch["p_erase"]
        row["direct_conditional_z"] = ch["p_z"]
        rows.append(dict(bath=bath, **row))
        if example is None:
            example = bath, signal, details

    assert [r["bath"] for r in rows] == list(range(4000, 4049))
    refs = {(int(r["Bath"]), r["Arm"]): float(r["Check_time"])
            for r in read(data / "raw_round_channels_per_bath.csv") if float(r["T_clock"]) == 6}
    for r in rows:
        r["fit_time"] = refs[r["bath"], FIT]
        r["exact_model_time"] = refs[r["bath"], EXACT]
        r["fft_minus_fit"] = r["fft_time"] - r["fit_time"]
        r["tone_minus_fit"] = r["tone_time"] - r["fit_time"]
    with (args.out / "per_bath.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(upstream_commit=PIN, baths=len(rows), reference=FIT, grid_step=0.06,
                   calibration_points=len(TIME), calibration_dt=DT, calibration_duration=TIME[-1],
                   fft=stats(rows, "fft_time"), single_tone=stats(rows, "tone_time"),
                   exact_model=stats(rows, "exact_model_time"),
                   fft_matches_exact_model=int(sum(abs(r["fft_time"] - r["exact_model_time"]) < 1e-8 for r in rows)),
                   max_occupancy_identity_error=max(r["max_occupancy_identity_error"] for r in rows),
                   max_frame_phase_error_rad=max(r["frame_phase_error_rad"] for r in rows),
                   max_channel_errors={key:max(abs(r["direct_"+key]-r["fft_"+key]) for r in rows)
                                       for key in ("erasure","conditional_z")},
                   ranges={key:[min(r[key] for r in rows), max(r[key] for r in rows)]
                           for key in ("fit_time","tone_frequency","tone_amplitude","fft_return_magnitude","fft_erasure","fft_conditional_z")},
                   python=platform.python_version(), versions={s:version(s) for s in ("numpy","scipy","qutip","pandas","matplotlib")})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure(args.out, example, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



