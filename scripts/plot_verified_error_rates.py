"""Draw the report's two rate charts from an independent shot-count audit."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from display_stim_results import DECODERS, MEMORIES


def plot_comparison(pooled, output):
    labels = list(DECODERS.values())
    counts = [pooled["methods"][name]["failures"] for name in DECODERS]
    shots = pooled["shots"]
    rates = np.array(counts) / shots
    fig, ax = plt.subplots(figsize=(8, 2.35))
    fig.subplots_adjust(left=.40, right=.95, top=.70, bottom=.08)
    fig.text(.037, .88, "Logical errors by decoder", size=17, weight="bold")
    fig.text(.037, .75, f"Wrong logical predictions out of {shots:,} shared Stim shots", size=11)
    ax.barh(range(len(labels)), rates, height=.52, color=["#8da3af", "#8da3af", "#086d68"])
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(rates) * 1.7, .001))
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=11, length=0, pad=8)
    ax.spines[:].set_visible(False)
    for i, (count, rate) in enumerate(zip(counts, rates, strict=True)):
        ax.annotate(f"{rate:.2%}   ({count:,})", (rate, i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scenarios(cases, output):
    bases = sorted({row["basis"] for row in cases})
    distances = sorted({row["distance"] for row in cases})
    probabilities = sorted({row["p"] for row in cases})
    fractions = {(r["basis"], r["distance"], r["p"]): r["methods"]["frozen"]["failures"] / r["shots"]
                 for r in cases}
    if len(fractions) != len(cases):
        raise ValueError("Duplicate memory, distance and noise level in scenario rows")
    palette = LinearSegmentedColormap.from_list("errors", ["#f3f6f7", "#80bab2", "#005953"])
    scale = Normalize(vmin=0, vmax=max(fractions.values()) or 1)
    fig, axs = plt.subplots(1, len(bases), figsize=(4.2 * len(bases), 3.1), squeeze=False)
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.67, wspace=.28)
    fig.text(.035, .88, f"{DECODERS['frozen']} logical error rate", size=17, weight="bold")
    shot_counts = {row["shots"] for row in cases}
    if len(shot_counts) == 1:
        fig.text(.035, .78, f"Each cell covers {next(iter(shot_counts)):,} shots.", size=10)
    for ax, basis in zip(axs[0], bases, strict=True):
        values = np.array([[fractions.get((basis, d, p), np.nan) for p in probabilities]
                           for d in distances])
        ax.imshow(values, cmap=palette, norm=scale, aspect="auto", interpolation="nearest")
        ax.set_title(MEMORIES.get(basis.lower(), f"{basis.upper()} memory"),
                     loc="left", size=12, weight="bold", pad=6)
        ax.set_xticks(range(len(probabilities)), [f"{100 * p:g}%" for p in probabilities])
        ax.set_yticks(range(len(distances)), [f"d = {d}" for d in distances])
        ax.tick_params(which="both", length=0, labelsize=10)
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(len(probabilities) - 1) + .5, minor=True)
        ax.set_yticks(np.arange(len(distances) - 1) + .5, minor=True)
        ax.grid(which="minor", color="white", lw=2)
        for (row, col), rate in np.ndenumerate(values):
            label = "0 seen" if rate == 0 else f"{rate:.2%}"
            ax.text(col, row, label if np.isfinite(rate) else "—", ha="center", va="center",
                    fontsize=11, color="white" if scale(rate) > .7 else "#152c31")
    fig.supxlabel("Physical error probability", y=.014, size=11)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="archive containing error-rate-audit.json")
    args = parser.parse_args()
    audit = json.loads((args.archive / "error-rate-audit.json").read_text(encoding="utf-8"))
    plot_comparison(audit["pooled"], args.archive / "verified-decoder-rates.png")
    plot_scenarios(audit["scenarios"], args.archive / "verified-scenario-rates.png")
