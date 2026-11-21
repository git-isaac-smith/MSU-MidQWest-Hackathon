"""Plot held-out paired logical-error reductions and local inference time by distance."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bascd.stats import paired_ler_difference

from display_stim_results import DECODERS


def plot(folder, output):
    policy = json.loads((folder / "frozen-policy.json").read_text())
    with (folder / "paired.csv").open(newline="") as src:
        pairs = [r for r in csv.DictReader(src) if r["reference"] == "bath_clock"]
    with (folder / "test.csv").open(newline="") as src:
        measurements = list(csv.DictReader(src))
    distances = sorted({int(r["distance"]) for r in pairs})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.45), layout="constrained")
    fig.suptitle("Logical error reduction and inference time", size=16, weight="bold")
    labels = [f"d = {d}" for d in distances]
    reference_times, selected_times = [], []
    for i, d in enumerate(distances):
        group = [r for r in pairs if int(r["distance"]) == d]
        result = paired_ler_difference(*zip(*[(int(r["shots"]), int(r["recovered"]), int(r["lost"]))
                                              for r in group]))
        gain = -100 * result["ler_difference"]
        interval = result["approximate_95_ci"]
        err = None
        if interval:
            lo, hi = -100 * interval[1], -100 * interval[0]
            err = [[gain-lo], [hi-gain]]
        left.errorbar(i, gain, yerr=err, fmt="o", capsize=5, color="#146951", markersize=8)
        subset = [r for r in measurements if int(r["distance"]) == d]
        for name, target in (("bath_clock", reference_times), (policy["by_distance"][str(d)], selected_times)):
            rows = [r for r in subset if r["decoder"] == name]
            target.append(1e6 * sum(float(r["median_batch_s"]) for r in rows) / sum(int(r["shots"]) for r in rows))
    left.axhline(0, color="#777", linewidth=1)
    left.set_title(f"Reduction versus {DECODERS['bath_clock'].lower()}", loc="left", fontsize=12)
    left.set_ylabel("Logical error rate reduction (percentage points)")
    left.set_xticks(range(len(distances)), labels)
    left.set_xlim(-.4, len(distances)-.6)
    x = np.arange(len(distances))
    right.bar(x-.19, reference_times, .36, color="#64758c", label=DECODERS["bath_clock"])
    right.bar(x+.19, selected_times, .36, color="#146951", label=DECODERS["frozen"])
    right.set_yscale("log")
    right.set_xticks(x, labels)
    right.set_title("Batch inference per shot", loc="left", fontsize=12)
    right.set_ylabel("Microseconds (log scale)")
    right.legend(frameon=False)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.17)
    with output.open("xb") as dest:
        fig.savefig(dest, format="png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.folder, args.output)
