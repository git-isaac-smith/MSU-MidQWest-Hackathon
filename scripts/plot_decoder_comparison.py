"""Plot pooled logical error rates and paired outcomes from a Stim shot archive."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from display_stim_results import DECODERS


def plot(archive, output):
    summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
    shots = summary["shots"]
    methods = summary["methods"]
    failures = [methods[name]["failures"] for name in DECODERS]
    rates = [100 * count / shots for count in failures]
    pair = summary["paired"]["native"]
    recovered, lost = pair["recovered"], pair["lost"]
    if recovered - lost != methods["native"]["failures"] - methods["frozen"]["failures"]:
        raise ValueError("Paired outcomes disagree with the pooled failure counts")

    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    fig.subplots_adjust(left=.33, right=.96, bottom=.32, top=.76)
    fig.text(.035, .90, "Logical error rate by decoder", weight="bold", size=19)
    fig.text(.035, .82, f"{shots:,} shared Stim shots across {summary['scenario_count']} scenarios",
             size=11, color="#415363")
    bars = ax.barh(list(DECODERS.values()), rates, height=.64,
                   color=["#4f6680", "#687f99", "#13745d"])
    for bar, count, rate in zip(bars, failures, rates, strict=True):
        ax.annotate(f"{rate:.4f}%  ({count:,})",
                    (rate, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.57 * max(rates) if any(rates) else 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=1))
    ax.set_xlabel("Logical error rate", labelpad=6)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    net = recovered - lost
    if net:
        comparison = f"{abs(net):,} {'fewer' if net > 0 else 'more'} wrong logical predictions"
        note = f"The hybrid decoder made {comparison} than standard matching."
    else:
        note = "The hybrid decoder and standard matching made the same number of logical errors."
    fig.text(.035, .125, note, weight="bold", size=11)
    fig.text(.035, .057, f"It corrected {recovered:,} shots standard matching missed and missed "
             f"{lost:,} shots standard matching corrected.", size=10, color="#415363")
    with output.open("xb") as dest:
        fig.savefig(dest, format="png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.archive, args.output)
