"""Render the frontier benchmark's completed, paired measurements."""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = {
    "mwpm": "Ordinary MWPM",
    "correlated": "Correlated MWPM",
    "union_find": "Union-find",
    "fast_hybrid": "Fast hybrid",
    "css_routed": "CSS routed",
    "temporal_lowpass": "Temporal low-pass",
    "local_peel": "Local peel + MWPM",
}
COLORS = ["#151c2c", "#005ba9", "#ca542c", "#507d13", "#824aae", "#a76800", "#008583"]
DISTANCE_COLORS = {3: "#005ba9", 5: "#c85d25", 7: "#148066"}


def _load_metrics(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("distance", "rounds", "shots", "errors", "fixed", "added"):
            row[field] = int(row[field])
        for field in (
            "p", "ler_percent", "wilson_low", "wilson_high", "batch_us_shot",
            "batch_us_round", "single_median_us", "single_p95_us", "setup_s", "paired_p",
        ):
            row[field] = float(row[field]) if row[field] else math.nan
    return rows


def _sweep_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["study"] == "sweep" and row["distance"] == row["rounds"]:
            groups[row["basis"], row["variant"], row["method"], row["distance"]].append(row)
    return {key: sorted(group, key=lambda row: row["p"]) for key, group in groups.items()}


def _crossings(groups):
    records = []
    for basis, variant, method in sorted({key[:3] for key in groups}):
        for smaller, larger in ((3, 5), (5, 7)):
            small = {row["p"]: row for row in groups.get((basis, variant, method, smaller), ())}
            large = {row["p"]: row for row in groups.get((basis, variant, method, larger), ())}
            points = []
            for p in sorted(small.keys() & large.keys()):
                a, b = small[p], large[p]
                delta = b["ler_percent"] - a["ler_percent"]
                certain = int(b["wilson_low"] > a["wilson_high"]) - int(b["wilson_high"] < a["wilson_low"])
                points.append((p, delta, certain))

            record = dict(basis=basis, variant=variant, method=method, smaller_distance=smaller,
                          larger_distance=larger, common_points=len(points))
            found = False
            for kind, column in (("point_estimate", 1), ("wilson_supported", 2)):
                nonzero = [point for point in points if point[column]]
                for left, right in zip(nonzero, nonzero[1:]):
                    if (left[column] < 0) == (right[column] < 0):
                        continue
                    estimated = ""
                    if kind == "point_estimate":
                        if right[0] <= 0 or left[0] <= 0:
                            estimated = left[0] - left[1] * (right[0] - left[0]) / (right[1] - left[1])
                        else:
                            fraction = -left[1] / (right[1] - left[1])
                            estimated = math.exp(math.log(left[0]) + fraction * math.log(right[0] / left[0]))
                    records.append(dict(
                        record, kind=kind, p_lower=left[0], p_upper=right[0], interpolated_p=estimated,
                        delta_lower_pp=left[1], delta_upper_pp=right[1],
                    ))
                    found = True
            if not found:
                records.append(dict(record, kind="no_observed_reversal"))
    return records


def _plot_thresholds(groups, out):
    for basis in sorted({key[0] for key in groups}):
        for variant in sorted({key[1] for key in groups if key[0] == basis}):
            methods = [method for method in METHODS if any(
                key[:3] == (basis, variant, method) for key in groups
            )]
            figure, axes = plt.subplots(math.ceil(len(methods) / 2), 2, figsize=(11, 10),
                                        sharex=True, sharey=True, layout="constrained")
            for ax, method in zip(axes.flat, methods):
                for distance, color in DISTANCE_COLORS.items():
                    rows = groups.get((basis, variant, method, distance), ())
                    if not rows:
                        continue
                    x = [row["p"] for row in rows]
                    ax.plot(x, [row["ler_percent"] for row in rows], "o-", ms=4, lw=1.6,
                            color=color, label=f"d = {distance}")
                    ax.fill_between(x, [row["wilson_low"] for row in rows],
                                    [row["wilson_high"] for row in rows], color=color, alpha=0.15)
                ax.set_title(METHODS[method], loc="left")
                ax.grid(alpha=0.22)
                ax.yaxis.set_major_formatter(PercentFormatter())
            for ax in list(axes.flat)[len(methods):]:
                ax.axis("off")
            axes.flat[0].legend(frameon=False, loc="upper left")
            if all(row["p"] > 0 for key, seq in groups.items() if key[:2] == (basis, variant) for row in seq):
                axes.flat[0].set_xscale("log")
            figure.suptitle(f"{basis.upper()} memory · {variant} · Finite-distance noise sweep",
                            fontsize=15, fontweight="bold")
            figure.supxlabel("Physical noise probability p")
            figure.supylabel("Logical error per memory experiment")
            figure.savefig(out / f"threshold-{basis}-{variant}.png", dpi=170)
            plt.close(figure)


def _plot_comparison(rows, out):
    selected = {}
    for row in rows:
        if row["basis"] != "z" or not math.isclose(row["p"], 0.005):
            continue
        if (row["distance"] == 3 and row["study"] == "primary") or (
            row["distance"] in (5, 7) and row["study"] == "sweep" and row["rounds"] == row["distance"]
        ):
            selected.setdefault(row["case"], {})[row["method"]] = row
    ordered = sorted(selected.values(), key=lambda group: (
        next(iter(group.values()))["distance"], next(iter(group.values()))["case"]
    ))
    for study, filename in (("primary", "comparison-primary.png"), ("sweep", "comparison-sweep.png")):
        cases = [group for group in ordered if next(iter(group.values()))["study"] == study]
        if cases:
            _plot_comparison_cases(cases, out / filename)


def _plot_comparison_cases(cases, path):
    figure, axes = plt.subplots(len(cases), 2, figsize=(11, 2.8 * len(cases) + 0.8),
                                squeeze=False, layout="constrained")
    for (left, right), group in zip(axes, cases):
        case = next(iter(group.values()))
        for index, method in enumerate(METHODS):
            row = group.get(method)
            if row is None:
                continue
            rate = row["ler_percent"]
            bounds = [[max(0, rate - row["wilson_low"])], [max(0, row["wilson_high"] - rate)]]
            left.errorbar(rate, index, xerr=bounds, fmt="o", color=COLORS[index], capsize=3)
            right.scatter(row["batch_us_shot"], index, color=COLORS[index], s=30)
        title = f"d = {case['distance']} · {case['study']} · {case['variant']}"
        left.set_title(title, loc="left", fontsize=11)
        left.set_yticks(range(len(METHODS)), METHODS.values())
        right.set_yticks(range(len(METHODS)), [])
        right.set_xscale("log")
        left.xaxis.set_major_formatter(PercentFormatter())
        for ax in (left, right):
            ax.set_ylim(len(METHODS) - 0.55, -0.45)
            ax.grid(axis="x", alpha=0.23)
    axes[-1, 0].set_xlabel("Logical error per memory experiment (95% Wilson interval)")
    axes[-1, 1].set_xlabel("Offline batch time (µs/shot, log scale)")
    figure.suptitle("Z memory at p = 0.005", fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _summary(rows, crossings):
    text = [
        "# Circuit-level decoder results",
        "",
        "Logical error rates are per complete memory experiment. Batch timing is measured in microseconds "
        "per shot. The per-round batch metric in the CSV divides that time by the number of rounds "
        "and describes offline throughput. Paired tests compare logical outcomes on the same detector shots.",
        "",
        "## Primary held-out case",
        "",
        "The primary hypothesis compares fast hybrid with ordinary MWPM on the Z memory "
        "at distance 3 and p = 0.005. Success requires fewer errors with two-sided exact "
        "McNemar p < 0.05 and lower median time for warmed, interleaved batches. Other decoders "
        "in the primary table are exploratory.",
        "",
        "| Case | Decoder | Errors / shots | LER | Batch µs/shot | Fixed / added vs MWPM | Paired p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["study"] == "primary":
            text.append(
                f"| {row['case']} | {METHODS[row['method']]} | {row['errors']:,} / {row['shots']:,} "
                f"| {row['ler_percent']:.4f}% | {row['batch_us_shot']:.3f} "
                f"| {row['fixed']:,} / {row['added']:,} | {row['paired_p']:.3g} |"
            )
    raw = {row["case"]: row for row in rows if row["method"] == "mwpm"}
    for row in rows:
        if row["study"] != "primary" or row["method"] != "fast_hybrid":
            continue
        baseline = raw[row["case"]]
        speedup = baseline["batch_us_shot"] / row["batch_us_shot"]
        passed = row["paired_p"] < .05 and row["fixed"] > row["added"] and speedup > 1
        verdict = "met" if passed else "fell short of"
        text.extend([
            "",
            f"Fast hybrid {verdict} the joint primary criterion with {row['fixed']:,} fixed, "
            f"{row['added']:,} added, p = {row['paired_p']:.3g}, and an MWPM-to-hybrid "
            f"batch runtime ratio of {speedup:.2f}×.",
        ])

    confirmations = [row for row in rows if row["study"] == "confirmation" and row["method"] == "css_routed"]
    if confirmations:
        text.extend([
            "", "## Secondary held-out confirmations", "",
            "The prespecified secondary comparisons evaluate CSS routed against ordinary MWPM "
            "at distances 5 and 7 in Z memory with p = 0.005. The Bonferroni-adjusted cutoff "
            "is p < 0.025 for each two-sided exact paired test. Joint success also requires a "
            "lower measured median batch time.",
            "",
            "| Distance | Errors / shots, CSS vs MWPM | Change in LER | Fixed / added | Paired p "
            "| Batch µs/shot, CSS vs MWPM | MWPM / CSS time | Joint result |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in sorted(confirmations, key=lambda row: row["distance"]):
            baseline = raw[row["case"]]
            delta = row["ler_percent"] - baseline["ler_percent"]
            speedup = baseline["batch_us_shot"] / row["batch_us_shot"]
            passed = row["paired_p"] < .025 and row["fixed"] > row["added"] and speedup > 1
            verdict = "Met both criteria" if passed else "Fell short"
            text.append(
                f"| {row['distance']} | {row['errors']:,} vs {baseline['errors']:,} / {row['shots']:,} "
                f"| {delta:+.4f} pp | {row['fixed']:,} / {row['added']:,} | {row['paired_p']:.3g} "
                f"| {row['batch_us_shot']:.3f} vs {baseline['batch_us_shot']:.3f} "
                f"| {speedup:.2f}× | {verdict} |"
            )

    text.extend([
        "", "## Exploratory wins", "",
        "The table screens the remaining comparisons for lower logical error "
        "(two-sided paired p < 0.05) and lower measured batch time than ordinary MWPM. "
        "The screen uses unadjusted p-values.",
        "",
    ])
    wins = []
    for row in rows:
        if row["study"] == "primary" and row["method"] == "fast_hybrid":
            continue
        if row["study"] == "confirmation" and row["method"] == "css_routed":
            continue
        baseline = raw.get(row["case"])
        if baseline and row["fixed"] > row["added"] and row["paired_p"] < .05:
            if row["batch_us_shot"] < baseline["batch_us_shot"]:
                wins.append((row, baseline))
    if wins:
        text.extend([
            "| Study | Case | Decoder | LER vs MWPM | Speedup | Paired p |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row, baseline in wins:
            speed = baseline["batch_us_shot"] / row["batch_us_shot"]
            text.append(
                f"| {row['study']} | {row['case']} | {METHODS[row['method']]} "
                f"| {row['ler_percent']:.4f}% vs {baseline['ler_percent']:.4f}% "
                f"| {speed:.2f}× | {row['paired_p']:.3g} |"
            )
    else:
        text.append("No exploratory candidate met both criteria.")

    text.extend([
        "", "## Stress cases", "",
        "| Case | Decoder | LER | Batch µs/shot | Fixed / added vs MWPM |",
        "|---|---|---:|---:|---:|",
    ])
    for row in rows:
        if row["study"] == "stress":
            text.append(
                f"| {row['case']} | {METHODS[row['method']]} | {row['ler_percent']:.4f}% "
                f"| {row['batch_us_shot']:.3f} | {row['fixed']:,} / {row['added']:,} |"
            )
    point = sum(record["kind"] == "point_estimate" for record in crossings)
    supported = sum(record["kind"] == "wilson_supported" for record in crossings)
    text.extend([
        "", "## Finite-distance crossings", "",
        f"The sweep yielded {point} point-estimate reversals and {supported} Wilson-supported brackets "
        "for adjacent distances. See [crossings.csv](crossings.csv) for physical-noise intervals.",
        "",
        "Point crossings interpolate the difference in per-memory logical error between sampled noise "
        "probabilities on the log-p axis. A Wilson-supported bracket requires disjoint 95% Wilson "
        "intervals with opposite order at its two ends. The bracket is descriptive and the study "
        "measures only finite distances with rounds equal to distance.", "",
    ])
    return "\n".join(text)


def write_report(out: Path) -> None:
    """Write the plots, numerical crossings, and Markdown summary."""
    out = Path(out)
    rows = _load_metrics(out / "metrics.csv")
    groups = _sweep_groups(rows)
    crossings = _crossings(groups)
    fields = [
        "basis", "variant", "method", "smaller_distance", "larger_distance", "common_points",
        "kind", "p_lower", "p_upper", "interpolated_p", "delta_lower_pp", "delta_upper_pp",
    ]
    with (out / "crossings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(crossings)
    _plot_thresholds(groups, out)
    _plot_comparison(rows, out)
    _plot_decision_table(rows, out)
    _plot_diagnostics(out)
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    with (out / "pareto.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "method", "errors", "shots", "ler_percent",
                                                    "batch_us_shot", "batch_us_round"])
        writer.writeheader()
        for group in by_case.values():
            for row in group:
                dominated = any(
                    other["errors"] <= row["errors"] and other["batch_us_shot"] <= row["batch_us_shot"]
                    and (other["errors"] < row["errors"] or other["batch_us_shot"] < row["batch_us_shot"])
                    for other in group
                )
                if not dominated:
                    writer.writerow({key: row[key] for key in writer.fieldnames})
    (out / "results.md").write_text(_summary(rows, crossings), encoding="utf-8")


def _plot_decision_table(rows, out):
    """Place error percentages and measured time side by side for the three locked comparisons."""
    lookup = {(r["distance"], r["method"]): r for r in rows if r["study"] in ("primary", "confirmation")}
    if (7, "css_routed") not in lookup:
        return
    fig = plt.figure(figsize=(12.2, 7.4), facecolor="white")
    ink, muted, teal = "#182c34", "#435b65", "#006b61"
    fig.text(.05, .925, "Logical error and decoder runtime", color=ink, fontsize=24, weight="bold")
    fig.text(.05, .877, "Z memory · 0.5% circuit-level noise · 3, 5 and 7 syndrome-extraction rounds",
             color=muted, fontsize=11)
    for x, title in zip((.05, .435, .575, .725, .865),
                        ("Decoder", "Logical error", "Wrong / shots", "Batch µs/shot", "Batch µs/round")):
        fig.text(x, .806, title, color=muted, fontsize=10.8, weight="bold")
    for d, top in ((3, .741), (5, .508), (7, .275)):
        candidate = "fast_hybrid" if d == 3 else "css_routed"
        fig.text(.05, top, f"Distance {d}", fontsize=12.5, color=ink, weight="bold")
        names = (("mwpm", "Ordinary PyMatching"), (candidate, "Posterior lookup" if d == 3 else "Selective correlated matching"),
                 ("correlated", "Always use correlated matching"))
        for i, (method, label) in enumerate(names):
            row = lookup[d, method]
            y = top - .047 * (i + 1)
            color = teal if method == candidate else ink
            values = (label, f"{row['ler_percent']:.4f}%", f"{row['errors']:,} / {row['shots']:,}",
                      f"{row['batch_us_shot']:.3f}", f"{row['batch_us_round']:.3f}")
            for x, val in zip((.05, .435, .575, .725, .865), values):
                fig.text(x, y, val, color=color, fontsize=11.3, weight="bold" if method == candidate else "normal")
    fig.text(.05, .053, "Per-round time is offline batch throughput. Timings exclude Stim sampling and decoder setup.",
             color=muted, fontsize=10)
    fig.savefig(out / "error-and-runtime.png", dpi=150)
    plt.close(fig)


def _plot_diagnostics(out):
    labels = {"ordinary": "Ordinary", "final_round_measurement": "Last-round readout flips",
              "measurement_burst": "Middle-round readout burst", "spatial_correlated": "Correlated data faults"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.2), layout="constrained")
    found = False
    for variant, label in labels.items():
        study = "sweep" if variant == "ordinary" else "stress"
        case = out / "cases" / f"{study}-d7-z-p0.005-{variant}"
        if not (case / "temporal-power.csv").is_file():
            continue
        found = True
        with (case / "detector-time-mean.csv").open() as stream:
            groups = defaultdict(list)
            for row in csv.DictReader(stream):
                if row["data_readout_closure"] == "0":
                    groups[row["x"], row["y"]].append((int(float(row["round"])), float(row["event_probability"])))
        by_time = defaultdict(list)
        for trace in groups.values():
            if len(trace) == 7:
                for t, prob in trace:
                    by_time[t].append(prob)
        times = sorted(by_time)
        left.plot(times, [100 * sum(by_time[t]) / len(by_time[t]) for t in times], "o-", ms=4, label=label)
        with (case / "temporal-power.csv").open() as stream:
            spectrum = [r for r in csv.DictReader(stream) if int(r["trace_length"]) == 7
                        and float(r["frequency_cycles_per_round"]) > 0]
        right.plot([float(r["frequency_cycles_per_round"]) for r in spectrum],
                   [float(r["mean_trace_power"]) for r in spectrum], "o-", ms=4)
    if found:
        left.set(xlabel="Stim detector time", ylabel="Mean detector-event rate", title="Where in time the changes appear")
        left.yaxis.set_major_formatter(PercentFormatter())
        right.set(xlabel="Cycles per round", ylabel="Power of the mean trace", title="Three nonzero frequency bins")
        for ax in (left, right):
            ax.grid(alpha=.22)
        left.legend(fontsize=8, frameon=False)
        fig.suptitle("Distance-7 noise diagnostics on checks observed for all seven rounds")
        fig.savefig(out / "spectral-diagnostics.png", dpi=155)
    plt.close(fig)
