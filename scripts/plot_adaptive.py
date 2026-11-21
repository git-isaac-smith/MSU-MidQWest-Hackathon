"""Plot held-out adaptive decoder accuracy against measured batch runtime."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "mwpm": "Ordinary MWPM",
    "correlated_mwpm": "Correlated MWPM",
    "count": "Event count",
    "spacetime": "Adjacent rounds",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    report = json.loads((directory / "test.json").read_text(encoding="utf-8"))
    points = report["summary"]
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    for name, row in points.items():
        feature = name.split("-")[0]
        color = "#08766d" if feature in ("count", "spacetime") else "#434e6a"
        marker = {"count": "o", "spacetime": "s"}.get(feature, "D")
        x = row["batch_us_per_shot"]
        y = 100 * row["ler"]
        ax.scatter(x, y, color=color, marker=marker, s=70, label=NAMES[feature])
    ax.set(title="Held-out accuracy and batch runtime", xlabel="Batch inference (µs/shot)",
           ylabel="Logical error rate (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    ax.margins(x=0.13, y=0.18)
    fig.savefig(directory / "accuracy-runtime.png", dpi=160)


if __name__ == "__main__":
    main()
