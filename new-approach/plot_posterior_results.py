"""Draw the held-out distance-five and distance-seven decoder comparison."""

import argparse
import json
from pathlib import Path
from textwrap import fill

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory


RESULTS = Path(__file__).resolve().parent / "results"
FIVE = ("mwpm", "balanced", "accuracy", "full_map")
SEVEN = ("mwpm", "selected", "aggressive")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _signed(number):
    return f"{number:+.5f}".replace("-", "−")


def _paired_text(difference, interval):
    low, high = interval
    return f"{_signed(difference)} pp\n[{_signed(low)}, {_signed(high)}]"


def plot(distance5, distance7, output, selected_label=None):
    """Plot observed logical errors and inference time from the same study."""
    five_data = _read(distance5)
    seven_data = _read(distance7)
    five = {row["mode"]: row for row in five_data["rows"]}
    seven = {row["mode"]: row for row in seven_data["rows"]}
    seven_modes = tuple(mode for mode in SEVEN if mode in seven)
    policy = Path(distance7).with_name("policy.json")
    cutoff = _read(policy)["cutoff"] if policy.is_file() else None
    if cutoff == 0.5:
        seven_modes = tuple(mode for mode in seven_modes if mode != "aggressive")
    # The cutoff-one deployment executes MWPM alone. The study also timed
    # routing and alternative corrections for its predeclared comparison arm.
    batch_us = {5: {mode: 1e6 * t["seconds"] / t["shots"] for mode, t in five_data.get("timings", {}).items()},
                7: {}}
    if seven_timing := seven_data.get("timings"):
        mwpm_seconds = sum(t["mwpm_seconds"] for t in seven_timing)
        routed_seconds = mwpm_seconds + sum(t["margin_seconds"] + t["posterior_seconds"] for t in seven_timing)
        seven_seconds = {"mwpm": mwpm_seconds, "aggressive": routed_seconds}
        if cutoff is not None or seven["selected"].get("queried", 0) > 0:
            seven_seconds["selected"] = mwpm_seconds if cutoff == 1 else routed_seconds
        batch_us[7] = {mode: 1e6 * seven_seconds[mode] / seven[mode]["shots"]
                       for mode in seven_modes if mode in seven_seconds}
    full_map_pair = next(pair for pair in five_data["paired_comparisons"]
                         if pair["candidate"] == "full_map" and pair["reference"] == "mwpm")
    pairs = {
        (5, "full_map"): _paired_text(full_map_pair["difference_percentage_points"],
                                    full_map_pair["paired_ci95_percentage_points"]),
        **{(7, mode): _paired_text(seven[mode]["difference_pp"], seven[mode]["paired95_pp"])
           for mode in seven_modes if mode != "mwpm"},
    }
    selected = seven["selected"]
    default_selected = "Development-selected posterior" if selected.get("queried", 1) else "Development-selected MWPM"
    seven_labels = {
        "mwpm": "PyMatching (MWPM)",
        "selected": selected_label or selected.get("label", default_selected),
        "aggressive": "Routed posterior (0.50 cutoff)",
    }

    ink, muted, grid = "#172931", "#536672", "#dbe5eb"
    blue, orange = "#145d8a", "#a44324"
    figure = Figure(figsize=(13.5, 7.2), facecolor="white")
    ax = figure.add_axes((0.245, 0.185, 0.196, 0.635))
    figure_x = blended_transform_factory(figure.transFigure, ax.transData)
    upper = max(row["wilson95_percent"][1] for row in (*five.values(), *seven.values()))
    ax.set(xlim=(0, min(100, upper * 1.13)), ylim=(0.25, 7.75), yticks=[])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax.tick_params(axis="x", length=0, pad=8, colors=muted, labelsize=10)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for name in ("left", "top", "right"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color(grid)

    ax.plot((0.045, 0.955), (3.72, 3.72), color=grid, linewidth=0.9,
            transform=figure_x, clip_on=False)
    seven_positions = (2.48, 1.70, 0.92) if len(seven_modes) == 3 else (2.30, 1.20)
    for distance, data, modes, positions, section_y, color in (
        (5, five, FIVE, (6.67, 5.89, 5.11, 4.33), 7.50, blue),
        (7, seven, seven_modes, seven_positions, 3.30, orange),
    ):
        ax.text(0.045, section_y, f"Distance {distance}", transform=figure_x, weight="bold", fontsize=14,
                va="center", color=color)
        for mode, y in zip(modes, positions, strict=True):
            row = data[mode]
            n = row["total" if distance == 5 else "shots"]
            rate = row["error_percent"]
            low, high = row["wilson95_percent"]
            marker_color = muted if mode == "mwpm" else color
            label = row["label"] if distance == 5 else seven_labels[mode]
            ax.errorbar(rate, y, xerr=[[max(0, rate - low)], [max(0, high - rate)]], fmt="o",
                        color=marker_color, markersize=5.5, capsize=3.5, elinewidth=1.4, zorder=3)
            ax.text(0.045, y, fill(label, width=27), transform=figure_x, va="center", color=ink, fontsize=11.5)
            ax.text(0.465, y, f"{rate:.5f}%", transform=figure_x, va="center", color=ink, fontsize=11)
            runtime = batch_us[distance].get(mode)
            ax.text(0.560, y, "—" if runtime is None else f"{runtime:.2f} µs", transform=figure_x,
                    va="center", color=ink, fontsize=11)
            ax.text(0.678, y, f'{row["errors"]:,} / {n:,}', transform=figure_x, va="center", color=muted, fontsize=10.5)
            if (distance, mode) in pairs:
                ax.text(0.822, y, pairs[distance, mode], transform=figure_x, va="center", color=marker_color,
                        fontsize=9.5)

    for x, heading in ((0.245, "Logical error"), (0.465, "Error rate"), (0.560, "Batch time / shot"),
                       (0.678, "Wrong / shots"), (0.822, "Change vs MWPM\n[95% paired CI]")):
        ax.text(x, 8.00, heading, transform=figure_x, color=muted, va="center", weight="bold", fontsize=10.25)

    figure.text(0.045, 0.925, "Logical error and decoder runtime", color=ink, weight="bold", fontsize=24)
    figure.text(0.045, 0.881, "Five rounds of independent noise · 0.8% data X faults · 4% readout faults",
                color=muted, fontsize=12)
    figure.text(0.045, 0.087, "Batch time is decoding time per shot in microseconds. It excludes setup and Stim sampling.",
                color=muted, fontsize=10.5)
    figure.text(0.045, 0.054, "Whiskers show 95% Wilson intervals. A negative paired change favors the candidate.",
                color=muted, fontsize=10.5)
    figure.savefig(output, dpi=170)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance5", type=Path, default=RESULTS / "full-map-01" / "results.json")
    parser.add_argument("--distance7", type=Path, default=RESULTS / "distance7-01" / "results.json")
    parser.add_argument("--selected-label", help="replace the distance-seven selected policy label")
    parser.add_argument("--out", type=Path, default=RESULTS / "distance7-01" / "decoder-comparison.png")
    args = parser.parse_args()
    plot(args.distance5, args.distance7, args.out, args.selected_label)
