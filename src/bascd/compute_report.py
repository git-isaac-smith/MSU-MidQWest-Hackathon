"""Render the saved compute campaign tables as Markdown and a PNG."""

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
from statistics import median

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator


METHOD_LABELS = {
    "ordinary": "Ordinary matching",
    "activity": "Activity routing",
    "correlated": "Correlated matching",
    "learned": "Learned routing",
    "learned_rescue": "Learned routing + rescue",
    "gap2": "Logical-gap routing",
    "disagreement_rescue": "Disagreement rescue",
    "belief": "Belief matching",
    "random_control": "Matched-budget random control",
}
_STYLES = dict(zip(
    METHOD_LABELS,
    [("#52525b", "o"), ("#b45309", "s"), ("#1d4ed8", "D"),
     ("#047857", "P"), ("#9333ea", "^"), ("#0e7490", "v"),
     ("#be123c", "X"), ("#a16207", "*"), ("#71717a", "h")],
    strict=True,
))


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No saved observations in {path}")
    return rows


def _by_repetition(rows, *keys):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        values = grouped.setdefault(key, {})
        repetition = int(row["repetition"])
        if repetition in values:
            raise ValueError(f"Duplicate timing repetition {repetition} for {key}")
        cost = float(row["microseconds_per_shot"])
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError(f"Invalid timing for {key}, repetition {repetition}")
        values[repetition] = cost, int(row["shots"])
    return grouped


def _median_time(values):
    return median(cost for cost, _ in values.values())


def _paired_ratios(baseline, candidate):
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired timing comparisons must use identical repetitions")
    ratios = []
    for repetition in sorted(baseline):
        reference, reference_shots = baseline[repetition]
        timed, timed_shots = candidate[repetition]
        if reference_shots != timed_shots:
            raise ValueError("Paired timing comparisons must use identical batch sizes")
        ratios.append(reference / timed)
    return np.array(ratios)


def _bootstrap_median(ratios):
    draws = np.random.default_rng(19).choice(ratios, size=(20000, len(ratios)))
    return np.quantile(np.median(draws, axis=1), [.025, .975])


def _configure_log_ticks(axis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_locator(NullLocator())


def _paired_row(row):
    delta = float(row["change_pp"])
    low, high = row["ci_low_pp"], row["ci_high_pp"]
    interval = f"[{float(low):.4f}, {float(high):.4f}]" if low and high else "n/a"
    return (f'| {row["distance"]} | {METHOD_LABELS[row["method"]]} | {METHOD_LABELS[row["baseline"]]} '
            f'| {int(row["recovered"]):,} | {int(row["introduced"]):,} '
            f'| {delta:+.4f} | {interval} | {float(row["paired_p"]):.3g} |')


def _summary(protocol, accuracy, pairs, timings, setup, scaling, routes, image_name):
    distances = [str(distance) for distance in protocol["distances"]]
    methods = protocol["methods"]
    smoke = protocol["smoke"]
    lines = [
        "# Circuit-level decoder compute study",
        "",
        f"The campaign uses {protocol['model']}. Both distance cases use the same model family and one logical observable.",
        "",
    ]
    if smoke:
        lines += ["**Smoke run.** The small sample checks report and decoder operation.", ""]
    training = " and ".join(
        f"d{d} used {len(protocol['training_seeds'][d]) * protocol['training_shots_per_seed']:,} shots"
        for d in distances
    )
    lines += [
        f"Training and scoring used separate sampler seeds. For the learned policy, {training}. "
        f"Each distance used {protocol['test_shots_per_seed']:,} test shots per seed across "
        f"{protocol['test_seeds_per_distance']} held-out seeds. [Protocol](protocol.json)",
        "",
        f"![Logical error versus serial batch throughput by distance and persistent-process scaling]({image_name})",
        "",
        "## Accuracy and serial decoding",
        "",
        "Logical failure is measured per complete memory shot. Batch throughput is reported as median wall time "
        "per shot on warmed, interleaved serial batches. Single-shot latency is the median duration of a "
        "warmed call on one complete detector record. Setup is measured separately from both.",
        "",
    ]
    for d in distances:
        lines += [
            f"### Distance {d}",
            "",
            "| Method | Failures / shots | Logical error, % (95% Wilson interval) | Batch, µs/shot | Single, µs | Setup, s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for method in (*methods, "random_control"):
            a = accuracy[d, method]
            batch = timings.get((d, method, "batch"))
            single = timings.get((d, method, "single"))
            serial = (f'{_median_time(batch):.3f} | {_median_time(single):.3f} | {setup[d, method]:.4f}'
                      if batch is not None else "n/a | n/a | n/a")
            lines.append(
                f'| {METHOD_LABELS[method]} | {int(a["errors"]):,} / {int(a["shots"]):,} | '
                f'{float(a["error_percent"]):.4f} [{float(a["wilson_low_percent"]):.4f}, '
                f'{float(a["wilson_high_percent"]):.4f}] | {serial} |'
            )
        shots = {n for _, n in timings[d, "ordinary", "batch"].values()}
        if len(shots) != 1:
            raise ValueError(f"Varying serial batch sizes at distance {d}")
        repeats = len(timings[d, "ordinary", "batch"])
        lines += ["", f"Serial batch measurements used {shots.pop():,} shots and {repeats} repetitions.", ""]
    lines += [
        "The matched-budget control used the same number of correlated calls per seed as learned routing. "
        "Its calls targeted uniformly sampled shots within each seed. "
        "[Accuracy table](accuracy.csv) · [Serial timing](timing.csv) · [Setup](setup.csv)",
        "",
        "## Paired accuracy",
        "",
        "The four primary comparisons use exact two-sided McNemar tests with a Bonferroni threshold of .0125 "
        "per comparison. Error changes are candidate minus baseline in percentage points. "
        "The approximate paired 95% intervals are unadjusted. The saved analysis omits intervals for fewer "
        "than 30 total disagreements.",
        "",
        "| Distance | Candidate | Baseline | Baseline errors recovered | Candidate errors introduced | Change, pp | Approx. 95% interval, pp | Exact p |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in distances:
        for method, baseline in (("learned", "activity"), ("disagreement_rescue", "correlated")):
            lines.append(_paired_row(pairs[d, method, baseline]))
    lines += [
        "",
        "Other comparisons are descriptive. [Paired accuracy table](paired.csv)",
        "",
        "## Paired serial timing for learned routing",
        "",
        "Speedup is the median of baseline / learned wall-time ratios on matching batch repetitions. "
        "Percentile bootstrap intervals resample the saved batch pairs 20,000 times using a fixed random seed. "
        "They describe timing variation on this host.",
        "",
        "| Distance | Baseline | Learned faster, batches | Median speedup | Bootstrap 95% interval |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for d in distances:
        for baseline in ("activity", "ordinary"):
            ratios = _paired_ratios(timings[d, baseline, "batch"], timings[d, "learned", "batch"])
            lo, hi = _bootstrap_median(ratios)
            lines.append(f"| {d} | {METHOD_LABELS[baseline]} | {np.count_nonzero(ratios > 1)} / "
                         f"{len(ratios)} | {median(ratios):.3f}× | [{lo:.3f}×, {hi:.3f}×] |")
    lines += [
        "",
        "## Routing work on held-out shots",
        "",
        "Rates pool the saved per-seed counts. Learned routing, activity routing, and logical-gap routing count "
        "correlated calls. Disagreement rescue and learned routing + rescue count belief calls. Full disagreement "
        "rescue evaluates correlated matching on every shot. Learned routing + rescue uses the learned correlated route.",
        "",
        "| Distance | Activity correlated, % | Learned correlated, % | Logical gap correlated, % | Full rescue belief, % | Learned + rescue belief, % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in distances:
        entries = routes[d]
        total = sum(int(row["shots"]) for row in entries)
        fields = ("activity", "learned", "gap2", "disagreement_rescue", "learned_rescue")
        rates = [f"{100 * sum(int(row[field]) for row in entries) / total:.3f}" for field in fields]
        lines.append(f"| {d} | " + " | ".join(rates) + " |")
    lines += [
        "",
        " · ".join(f"[Distance {d} route counts](d{d}/route-counts.csv)" for d in distances),
        "",
        "## Persistent-process scaling",
        "",
        "Each worker compiles its own decoder once. Worker timing includes NumPy transfer through multiprocessing "
        "pipes and ordered assembly. The timer starts after worker startup and warmup. Speedup compares "
        "paired worker-pool runs against that method’s own one-worker pool.",
        "",
        "| Distance | Method | Workers | Shots / batch | Repetitions | Median µs/shot | Speedup vs one worker | Pool setup, s | Pool warmup, s |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    parallel = _by_repetition(scaling, "distance", "method", "workers")
    startup = {(r["distance"], r["method"], r["workers"]): (float(r["setup_s"]), float(r["warmup_s"])) for r in scaling}
    for key, repetitions in sorted(parallel.items(), key=lambda item: (item[0][:2], int(item[0][2]))):
        d, method, workers = key
        speedup = median(_paired_ratios(parallel[d, method, "1"], repetitions))
        _, shots = next(iter(repetitions.values()))
        setup_s, warmup_s = startup[key]
        lines.append(f"| {d} | {METHOD_LABELS[method]} | {workers} | {shots:,} | {len(repetitions)} | "
                     f"{_median_time(repetitions):.3f} | {speedup:.3f}× | {setup_s:.4f} | {warmup_s:.4f} |")
    lines += [
        "",
        f"Host measurements were taken on {protocol['host']} with Python {protocol['python']}. "
        "Timing begins with the complete detector record available in host memory. "
        "[Scaling repetitions](scaling.csv) · [Environment and package versions](protocol.json)",
        "",
    ]
    return "\n".join(lines)


def _plot(protocol, accuracy, timing, scaling):
    distances = list(map(str, protocol["distances"]))
    methods = protocol["methods"]
    parallel = _by_repetition(scaling, "distance", "method", "workers")
    scale_distances = {d for d, _, _ in parallel}
    if len(scale_distances) != 1:
        raise ValueError("Expected scaling data at a single distance")
    d_scale = next(iter(scale_distances))
    fig = Figure(figsize=(15, 5.6), dpi=170)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(ncols=len(distances) + 1)
    fig.subplots_adjust(left=.065, right=.982, bottom=.27, top=.81, wspace=.38)
    fig.suptitle("Logical accuracy and decoding throughput", y=.968, fontsize=18, fontweight="bold")
    subtitle = "Provisional circuit-level noise · rotated Z memory · p = 0.005 · rounds = distance"
    if protocol["smoke"]:
        subtitle += " · smoke run"
    fig.text(.5, .898, subtitle, ha="center", color="#475569")
    for d, ax in zip(distances, axes):
        for method in methods:
            a = accuracy[d, method]
            point = float(a["error_percent"])
            ci = [float(a["wilson_low_percent"]), float(a["wilson_high_percent"])]
            color, symbol = _STYLES[method]
            ax.errorbar(_median_time(timing[d, method, "batch"]), point,
                        yerr=[[max(0., point - ci[0])], [max(0., ci[1] - point)]], color=color,
                        marker=symbol, linestyle="none", capsize=3, markersize=7)
        ax.set(title=f"Distance {d} · serial", xlabel="Median batch time (µs/shot)", ylabel="Logical error (%)",
               xscale="log")
        _configure_log_ticks(ax.xaxis)
        ax.set_ylim(bottom=0.)
        ax.grid(True, alpha=.23)
    ax = axes[-1]
    workers = sorted({int(w) for _, _, w in parallel})
    for method in methods:
        series = {int(w): _median_time(rows) for (d, m, w), rows in parallel.items() if m == method}
        if series:
            color, symbol = _STYLES[method]
            counts = sorted(series)
            ax.plot(counts, [series[w] for w in counts], color=color, marker=symbol, markersize=6)
    ax.set(title=f"Distance {d_scale} · process scaling", xlabel="Persistent worker processes",
           ylabel="Median batch time (µs/shot)", xscale="log", yscale="log")
    ax.set_xticks(workers, labels=workers)
    ax.xaxis.set_minor_locator(NullLocator())
    _configure_log_ticks(ax.yaxis)
    ax.grid(True, alpha=.23)
    handles = [Line2D([], [], color=_STYLES[m][0], marker=_STYLES[m][1],
                      linestyle="none", label=METHOD_LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .06))
    fig.text(.5, .04, "Serial error bars show 95% Wilson intervals. Scaling time includes process IPC.",
             ha="center", color="#475569", fontsize=9)
    buffer = BytesIO()
    canvas.print_png(buffer)
    return buffer.getvalue()


def write_report(folder: Path):
    """Read published campaign tables and create two new report files."""
    summary_path = folder / "summary.md"
    image_path = folder / "compute-frontier.png"
    for path in (summary_path, image_path):
        if path.exists():
            raise FileExistsError(f"Choose an output folder without an existing {path.name}")
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    accuracy = {(r["distance"], r["method"]): r for r in _read_csv(folder / "accuracy.csv")}
    pairs = {(r["distance"], r["method"], r["baseline"]): r for r in _read_csv(folder / "paired.csv")}
    timings = _by_repetition(_read_csv(folder / "timing.csv"), "distance", "method", "measurement")
    setup = {(r["distance"], r["method"]): float(r["setup_s"]) for r in _read_csv(folder / "setup.csv")}
    scaling = _read_csv(folder / "scaling.csv")
    routes = {str(d): _read_csv(folder / f"d{d}" / "route-counts.csv") for d in protocol["distances"]}
    markdown = _summary(protocol, accuracy, pairs, timings, setup, scaling, routes, image_path.name)
    png = _plot(protocol, accuracy, timings, scaling)
    with image_path.open("xb") as stream:
        stream.write(png)
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    return summary_path, image_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for artifact in write_report(args.out):
        print(artifact)


if __name__ == "__main__":
    main()
