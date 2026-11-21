"""Plot per-scenario logical error rates for the distance-based hybrid decoder."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from display_stim_results import DECODERS, MEMORIES


def plot(archive, output):
    with (archive / "stats.csv").open(encoding="utf-8", newline="") as src:
        rows = [r for r in csv.DictReader(src) if r["decoder"] == "frozen"]
    bases = sorted({r["basis"] for r in rows})
    fig, axes = plt.subplots(1, len(bases), sharey=True, squeeze=False,
                             figsize=(5 * len(bases), 4.4), layout="constrained")
    fig.suptitle(f"{DECODERS['frozen']} logical error rates", size=16, weight="bold")
    colors = {3: "#325e89", 5: "#b45626", 7: "#187457"}
    markers = {3: ("o", 6), 5: ("s", 8), 7: ("D", 4)}
    for ax, basis in zip(axes[0], bases, strict=True):
        selected = [r for r in rows if r["basis"] == basis]
        for d in sorted({int(r["distance"]) for r in selected}):
            group = sorted((r for r in selected if int(r["distance"]) == d), key=lambda r: float(r["p"]))
            x, y, low, high = (100 * np.array([float(r[k]) for r in group])
                               for k in ("p", "ler", "wilson95_low", "wilson95_high"))
            marker, size = markers[d]
            ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), color=colors[d],
                        marker=marker, markersize=size, markerfacecolor="white" if d == 5 else colors[d],
                        capsize=3, label=f"Distance {d}")
        ax.set_title(MEMORIES.get(basis.lower(), f"{basis.upper()} memory"), loc="left")
        ax.set_xlabel("Physical error probability (%)")
        ax.set_xticks(sorted({100 * float(r["p"]) for r in selected}))
        ax.set_yscale("symlog", linthresh=.04)
        ax.set_yticks([0, .04, .1, 1, 10], ["0", "0.04", "0.1", "1", "10"])
        ax.grid(axis="y", alpha=.2)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0, 0].set_ylabel("Logical error rate (%)")
    axes[0, 0].legend(frameon=False)
    with output.open("xb") as dest:
        fig.savefig(dest, format="png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.archive, args.output)
