"""Plot the first logical error by the distance-based hybrid decoder in an archived Stim case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import stim

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from display_stim_results import DECODERS
from verify_stim_samples import read_shots


def plot_syndrome(archive: Path, case_name: str) -> None:
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    case = next((case for case in manifest["cases"] if case["case"] == case_name), None)
    if case is None:
        raise ValueError(f"Case {case_name!r} is missing from the archive manifest")
    shots = manifest["shots_per_scenario"]
    folder = archive / "cases" / case_name
    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    if circuit.num_observables != 1:
        raise ValueError("Expected a single logical observable")
    observed = read_shots(folder / "observables.b8", shots=shots, bits=1)[:, 0]
    predicted = read_shots(folder / "frozen.b8", shots=shots, bits=1)[:, 0]
    failures = np.flatnonzero(observed != predicted)
    if not failures.size:
        raise ValueError(f"No logical errors by the {DECODERS['frozen'].lower()} in case {case_name!r}")
    shot = int(failures[0])
    detectors = read_shots(
        folder / "detectors.b8", shots=shots, bits=circuit.num_detectors, is_detector=True
    )
    fired = np.unpackbits(
        detectors[shot], count=circuit.num_detectors, bitorder="little"
    ).astype(bool)
    coordinates = circuit.get_detector_coordinates()
    xyz = np.array([coordinates[i] for i in range(circuit.num_detectors)])
    x, y, time = xyz.T
    times = np.unique(time)
    fired_color = "#b54224"
    quiet_color = "#8597a7"
    ink = "#193044"

    columns = min(4, len(times))
    rows = (len(times) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.1 * columns, 2.8 * rows + 1.8),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24 / rows, top=1 - 0.96 / (2.8 * rows + 1.8),
                        wspace=0.18, hspace=0.38)
    for ax, t in zip(axes.flat, times):
        on_slice = time == t
        quiet = on_slice & ~fired
        active = on_slice & fired
        ax.scatter(x[quiet], y[quiet], s=52, facecolors="white", edgecolors=quiet_color,
                   linewidths=1.5, zorder=3)
        ax.scatter(x[active], y[active], s=87, color=fired_color, zorder=4)
        for detector in np.flatnonzero(active):
            ax.annotate(f"D{detector}", (x[detector], y[detector]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, color=fired_color, fontweight="bold")
        ax.set_title(f"t = {t:g}  |  {active.sum()} fired", fontsize=11, color=ink, pad=8)
        ax.set(xlim=(x.min() - 0.85, x.max() + 0.85), ylim=(y.min() - 0.85, y.max() + 1.15),
               xticks=np.unique(x), yticks=np.unique(y), aspect="equal")
        ax.tick_params(axis="both", labelsize=8, length=0, colors=ink)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("x", color=ink, labelpad=1)
        ax.grid(color="#e1e8ee", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color("#d0dae3")
    for ax in axes.flat[len(times):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("y", color=ink)

    fig.text(0.07, 0.96, "Syndrome at a wrong logical prediction",
             fontsize=18, fontweight="bold", color=ink, va="top")
    fig.text(
        0.07, 0.96 - 0.33 / fig.get_figheight(),
        f"Distance {case['distance']}  /  {case['basis'].upper()} memory  /  "
        f"p = {case['p']:g}  /  shot index {shot} (zero-based)",
        fontsize=11, color=ink, va="top",
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=fired_color,
                   markeredgecolor=fired_color, label="Fired detector (1)"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=quiet_color, label="Quiet detector (0)"),
        ],
        loc="lower left", bbox_to_anchor=(0.064, 0.11 / rows), ncols=2, frameon=False,
    )
    fig.text(
        0.07, 0.085 / rows,
        f"Logical flip bit  |  observed {observed[shot]}  /  "
        f"{DECODERS['frozen'].lower()} predicted {predicted[shot]}",
        fontsize=11, fontweight="bold", color=ink,
    )
    fig.text(0.07, 0.043 / rows, "Panels show recorded detector checks at each Stim time coordinate.",
             fontsize=9, color=ink)
    output = archive / "syndrome-example.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    fired_ids = ", ".join(f"D{i}" for i in np.flatnonzero(fired)) or "none"
    print(
        f"{case_name} shot {shot} (zero-based): observed {observed[shot]}, "
        f"{DECODERS['frozen'].lower()} predicted {predicted[shot]}, fired {fired_ids}. Wrote {output}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--case", default="d3-z-p0.005", help="Case name from the archive manifest")
    args = parser.parse_args()
    plot_syndrome(args.archive, args.case)


if __name__ == "__main__":
    main()
