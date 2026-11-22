"""Plot measured logical error rates for a completed burst-sensitivity run."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


PANELS = {
    "poisson": "Fixed-density Poisson bursts",
    "one": "Exactly one burst per shot",
    "independent": "Independent marginal flips",
}
STYLES = [
    ("#1368a3", "o", "-"), ("#a14a19", "s", "--"),
    ("#187969", "^", "-."), ("#8a4a8f", "D", ":"),
]
CONTROL = ("#536074", "x", ":")


def _load_results(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No benchmark rows in {path}")
    for row in rows:
        for key in ("distance", "rounds", "shots", "errors"):
            row[key] = int(row[key])
        for key in (
            "background_p", "density", "xi", "ler_percent",
            "wilson_low_percent", "wilson_high_percent",
        ):
            row[key] = float(row[key])
        if row["exposure"] not in (*PANELS, "none"):
            raise ValueError(f"Unknown exposure {row['exposure']!r} in {path}")
    return rows


def _group_rows(rows):
    settings = defaultdict(lambda: defaultdict(list))
    background_controls = defaultdict(list)
    for row in rows:
        if row["exposure"] == "none":
            background_controls[row["background_p"]].append(row)
        else:
            settings[row["background_p"], row["density"]][row["exposure"], row["xi"]].append(row)
    for (background_p, density), series in settings.items():
        controls = background_controls[background_p]
        matching_controls = [row for row in controls if row["density"] == density]
        if matching_controls or controls:
            series["none", 0] = matching_controls or controls
    if not settings:
        raise ValueError("The run has no burst or independent-marginal rows")
    for series in settings.values():
        for points in series.values():
            points.sort(key=lambda row: row["distance"])
            if len({row["distance"] for row in points}) != len(points):
                raise ValueError("Multiple rows for a distance in the same plot series")
    return sorted(settings.items())


def _draw_series(axis, points, style):
    color, marker, linestyle = style
    rate = [row["ler_percent"] for row in points]
    lower = [max(0.0, value - row["wilson_low_percent"]) for value, row in zip(rate, points)]
    upper = [max(0.0, row["wilson_high_percent"] - value) for value, row in zip(rate, points)]
    axis.errorbar(
        [row["distance"] for row in points], rate, yerr=[lower, upper],
        color=color, marker=marker, linestyle=linestyle, linewidth=1.55,
        markersize=5, capsize=3, elinewidth=1.1,
    )


def _plot(settings, output_stem):
    all_xi = sorted({xi for _, series in settings for exposure, xi in series if exposure != "none"})
    styles = {xi: STYLES[index % len(STYLES)] for index, xi in enumerate(all_xi)}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#18253a"}):
        figure, axes = plt.subplots(
            len(settings), len(PANELS), figsize=(12, 2.25 + 2.65 * len(settings)),
            squeeze=False,
        )
        figure.subplots_adjust(left=0.09, right=0.98, top=1 - 1.23 / figure.get_figheight(),
                               bottom=0.82 / figure.get_figheight(), wspace=0.28, hspace=0.51)
        figure.text(0.09, 1 - 0.29 / figure.get_figheight(), "Logical errors under burst noise",
                    fontsize=20, fontweight="bold", va="top")
        subtitle = "Burst-blind MWPM  ·  95% Wilson intervals"
        if len(settings) == 1:
            (background_p, density), _ = settings[0]
            subtitle += f"  ·  Background p = {background_p:g}  ·  Poisson density = {density:g}"
        figure.text(0.09, 1 - 0.71 / figure.get_figheight(), subtitle, color="#536074", va="top")
        for ((background_p, density), series), row_axes in zip(settings, axes):
            distances = sorted({row["distance"] for points in series.values() for row in points})
            for (exposure, title), axis in zip(PANELS.items(), row_axes):
                axis.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
                if len(settings) > 1:
                    axis.set_title(f"{title}\nBackground p = {background_p:g}, Poisson density = {density:g}",
                                   fontsize=9, loc="left", pad=6)
                control = series.get(("none", 0))
                if control:
                    _draw_series(axis, control, CONTROL)
                for xi in all_xi:
                    if (exposure, xi) in series:
                        _draw_series(axis, series[exposure, xi], styles[xi])
                if not any(key[0] == exposure for key in series):
                    axis.text(0.5, 0.55, "No measurements", transform=axis.transAxes,
                              ha="center", color="#536074")
                axis.set_xticks(distances)
                axis.set_xlim(distances[0] - 0.45, distances[-1] + 0.45)
                comparable = {"none", "one"} if exposure == "one" else {"none", "poisson", "independent"}
                highest = max(row["wilson_high_percent"] for (group, _), points in series.items()
                              if group in comparable for row in points)
                axis.set_ylim(0, max(1.0, highest * 1.075))
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                axis.set_xlabel("Code distance")
                axis.grid(axis="y", color="#dce3ec", linewidth=0.8)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                for side in ("left", "bottom"):
                    axis.spines[side].set_color("#aeb9c9")
                axis.tick_params(color="#aeb9c9", labelcolor="#38475b")
            row_axes[0].set_ylabel("Logical error rate per shot")
        legend_entries = [(f"ξ = {xi:g}", styles[xi]) for xi in all_xi]
        if any(("none", 0) in series for _, series in settings):
            legend_entries.insert(0, ("No added bursts", CONTROL))
        handles = [Line2D([], [], color=color, marker=marker, linestyle=line, label=label)
                   for label, (color, marker, line) in legend_entries]
        figure.legend(handles=handles, loc="lower center", ncols=len(handles), frameon=False,
                      bbox_to_anchor=(0.5, 0.035 / figure.get_figheight()))
        figure.savefig(output_stem.with_suffix(".png"), dpi=220)
        figure.savefig(output_stem.with_suffix(".pdf"))
        plt.close(figure)


def _endpoint_summary(points):
    first, last = points[0], points[-1]
    if first is last:
        return "One distance measured"
    if first["wilson_high_percent"] < last["wilson_low_percent"]:
        return "Higher at largest distance, intervals separated"
    if last["wilson_high_percent"] < first["wilson_low_percent"]:
        return "Lower at largest distance, intervals separated"
    change = last["ler_percent"] - first["ler_percent"]
    direction = "Higher" if change > 0 else "Lower" if change < 0 else "Equal"
    return f"{direction} at largest distance, intervals overlap"


def _marginal_comparison(series):
    rows = []
    for (exposure, xi), poisson in sorted(series.items()):
        if exposure != "poisson" or ("independent", xi) not in series:
            continue
        controls = {row["distance"]: row for row in series["independent", xi]}
        for row in poisson:
            if row["distance"] in controls:
                rows.append((xi, row, controls[row["distance"]]))
    if not rows:
        return []
    lines = [
        "### Poisson bursts versus independent marginal flips", "",
        "| ξ | d | Poisson LER | Independent LER | Poisson minus independent (percentage points) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for xi, poisson, independent in rows:
        poisson_rate, independent_rate = poisson["ler_percent"], independent["ler_percent"]
        lines.append(f"| {xi:g} | {poisson['distance']} | {poisson_rate:.4g}% | "
                     f"{independent_rate:.4g}% | {poisson_rate - independent_rate:+.4g} |")
    return lines + [""]


def _write_report(path, settings, source, protocol):
    lines = [
        "# Spatial burst sensitivity",
        "",
        "![Logical error rates by code distance, exposure, and burst length scale](burst_sensitivity.png)",
        "",
        "Spatial bursts are injected on top of the baseline circuit noise. The benchmark uses "
        "burst-blind minimum-weight perfect matching (MWPM), with its decoder graph and weights "
        "from the baseline circuit. The no-burst control measures that baseline. The independent "
        "marginal control matches the single-site, single-round flip probabilities of the "
        "fixed-density Poisson model. Burst centers are chosen uniformly over the data sites. "
        "The length scale ξ is measured in nearest-neighbor data-qubit spacings. Poisson density "
        "is the expected number of burst cores per data qubit per syndrome round.",
        "",
        "Rates are logical failures per shot. Error bars show individual 95% Wilson intervals. "
        "The table describes the two measured endpoints for each series. Interval overlap is "
        "descriptive and does not constitute a test of the difference.",
        "",
    ]
    for (background_p, density), series in settings:
        lines.extend([
            f"## Background p = {background_p:g}, Poisson density = {density:g}", "",
            "| Exposure | ξ | Smallest d | Largest d | Change (percentage points) | Endpoint comparison |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for exposure in ("none", *PANELS):
            for (kind, xi), points in sorted(series.items()):
                if kind != exposure:
                    continue
                first, last = points[0], points[-1]
                delta = last["ler_percent"] - first["ler_percent"]
                smallest = f"{first['distance']} ({first['ler_percent']:.4g}%)"
                largest = f"{last['distance']} ({last['ler_percent']:.4g}%)"
                lines.append(
                    f"| {PANELS.get(kind, 'No added bursts')} | {f'{xi:g}' if xi else '—'} | {smallest} | "
                    f"{largest} | {delta:+.4g} | {_endpoint_summary(points)} |"
                )
        lines.append("")
        lines.extend(_marginal_comparison(series))
    lines.extend([
        "## Interpreting the exposure", "",
        "The single-burst flip probability at one decay length is 0.5/e, about 18.4%. "
        "This length measures spatial reach. The rate of burst starts is an independent model choice. "
        "With d² data sites and d rounds, the fixed-density model expects density × d³ burst cores per shot. "
        "The middle panel holds exposure at one burst per shot. Its vertical scale accommodates that higher exposure. "
        "The outside panels use the same vertical scale.", "",
        "The independent control uses the odd-flip probability at each site after averaging over core positions "
        "and counts. For site j this is (1 − exp(−2 × density × sum of kernel weights to j))/2 per round. "
        "This control preserves the mean local fault load while removing the shared core. "
        "The sweep measures a finite set of distances in a burst stress test with ordinary matching.", "",
        "## Reproduce the sweep", "",
        "For the default study, run from the repository root with the project Python environment. "
        "Choose a fresh output directory. The sweep accepts custom lengths, density, distances, shot counts, and seeds.", "",
        "```sh", "python -m bascd.burst_sweep --out results/my-spatial-burst --xi 1 2 4 --density 0.001",
        "python -m bascd.burst_report --run results/my-spatial-burst", "```", "",
        f"[Benchmark results]({source}) · [Run protocol]({protocol}) · [Figure PDF](burst_sensitivity.pdf)",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="directory containing results.csv and protocol.json")
    parser.add_argument("--csv", type=Path, help="override the path to results.csv")
    args = parser.parse_args(argv)
    source = args.csv or args.run / "results.csv"
    protocol = args.run / "protocol.json"
    with protocol.open(encoding="utf-8-sig") as stream:
        json.load(stream)
    settings = _group_rows(_load_results(source))
    _plot(settings, args.run / "burst_sensitivity")
    source_link = source.name if source.parent.resolve() == args.run.resolve() else source.resolve().as_uri()
    _write_report(args.run / "report.md", settings, source_link, protocol.name)


if __name__ == "__main__":
    main()
