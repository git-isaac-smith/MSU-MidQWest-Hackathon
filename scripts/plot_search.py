"""Plot a completed campaign from its archived validation measurements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bascd.search import pareto, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    directory = parser.parse_args().campaign
    trials = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    refs = json.loads((directory / "validation-references.json").read_text(encoding="utf-8"))
    frontier = set(pareto(trials))
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, color, label in [(False, "#657b86", "Other candidates"),
                                 (True, "#007a69", "Candidate frontier")]:
        group = [t for t in trials if t["status"] == "ok" and (t["trial"] in frontier) == status]
        group.sort(key=lambda t: t["validation"]["batch_us_per_shot"])
        x = [t["validation"]["batch_us_per_shot"] for t in group]
        y = [100 * t["validation"]["ler"] for t in group]
        ax.plot(x, y, marker="o", ls="-" if status else "none", color=color, label=label)
        for i, t in enumerate(group):
            ax.annotate(str(t["trial"]), (x[i], y[i]), xytext=(5, 5), textcoords="offset points")
    for kind, label, marker in [("mwpm", "MWPM", "s"), ("correlated_mwpm", "Correlated MWPM", "D")]:
        score = summarize([r for r in refs if r["decoder"] == kind])
        ax.scatter(score["batch_us_per_shot"], 100 * score["ler"], marker=marker,
                   color="#b45220", s=65, label=label)
    ax.set(xlabel="Batch inference (µs/shot)", ylabel="Logical error rate (%)",
           title="Validation on the equally weighted 24-scenario grid")
    ax.grid(alpha=0.2)
    ax.margins(y=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "validation-pareto.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
