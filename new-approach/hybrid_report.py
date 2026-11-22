"""Summarize paired logical-decoding decisions and plot their error rates."""

import csv
import json
from math import isfinite
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.stats import binomtest

from bascd.stats import paired_ler_difference, wilson95


LABELS = {
    "mwpm": "PyMatching (MWPM)",
    "raw_learned": "Raw checks + CNN",
    "learned": "FFT features + CNN",
    "margin_only": "Matching margin + exact",
    "balanced": "Balanced FFT hybrid",
    "accuracy": "Exact on ambiguous shots",
}


def _binary_vector(values, name, shape=None):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or (shape is not None and values.shape != shape):
        raise ValueError(f"{name} must contain one value per shot")
    if values.dtype.kind not in "buif" or np.any((values != 0) & (values != 1)):
        raise ValueError(f"{name} must contain binary values")
    return values.astype(bool, copy=False)


def _rates(failed, queried):
    total = len(failed)
    errors = int(failed.sum())
    exact_queries = int(queried.sum())
    return {
        "errors": errors,
        "total": total,
        "error_percent": 100 * errors / total,
        "wilson95_percent": [100 * bound for bound in wilson95(errors, total)],
        "exact_queries": exact_queries,
        "exact_route_percent": 100 * exact_queries / total,
    }


def _paired(candidate, reference, failures, groups):
    reference_failed = failures[reference]
    candidate_failed = failures[candidate]
    gains = reference_failed & ~candidate_failed
    losses = ~reference_failed & candidate_failed
    per_seed = [dict(seed=seed, total=int(mask.sum()), gains=int(gains[mask].sum()),
                     losses=int(losses[mask].sum())) for seed, mask in groups.items()]
    gained, lost = int(gains.sum()), int(losses.sum())
    total, discordant = len(gains), gained + lost

    if discordant >= 30 and all(row["total"] >= 2 for row in per_seed):
        stats = paired_ler_difference([row["total"] for row in per_seed],
                                      [row["gains"] for row in per_seed],
                                      [row["losses"] for row in per_seed])
        interval = stats["approximate_95_ci"]
        method = "Stratified paired normal approximation"
    else:
        # The exact marginal intervals jointly cover gains and losses with
        # at least 95% confidence by the union bound, even for sparse pairs.
        gain_ci = binomtest(gained, total).proportion_ci(confidence_level=0.975, method="exact")
        loss_ci = binomtest(lost, total).proportion_ci(confidence_level=0.975, method="exact")
        interval = loss_ci.low - gain_ci.high, loss_ci.high - gain_ci.low
        method = "Conservative exact binomial bounds"

    return {
        "candidate": candidate,
        "reference": reference,
        "total": total,
        "gains": gained,
        "losses": lost,
        "difference_percentage_points": 100 * (lost - gained) / total,
        "paired_ci95_percentage_points": [100 * bound for bound in interval],
        "paired_ci_method": method,
        "mcnemar_two_sided_p": binomtest(gained, discordant).pvalue if discordant else 1.0,
        "per_seed": per_seed,
    }


def _timing_rows(timings):
    result = {}
    for mode, values in timings.items():
        shots = int(values["shots"])
        seconds = float(values["seconds"])
        batch_size = int(values["batch_size"])
        if shots < 0 or not isfinite(seconds) or seconds < 0 or batch_size < 1:
            raise ValueError(f"invalid timing for {mode}")
        result[mode] = dict(shots=shots, seconds=seconds, batch_size=batch_size,
                            shots_per_second=shots / seconds if seconds else None)
    return result


def _write_csv(path, summary):
    def rate_fields(row):
        low, high = row["wilson95_percent"]
        return {key: row[key] for key in ("errors", "total", "error_percent", "exact_queries",
                                          "exact_route_percent")} | {"wilson95_low_percent": low,
                                                                     "wilson95_high_percent": high}

    fields = ("scope", "mode", "label", "seed", "errors", "total", "error_percent",
              "wilson95_low_percent", "wilson95_high_percent", "exact_queries", "exact_route_percent",
              "reference", "gains", "losses", "difference_percentage_points", "paired_ci95_low_pp",
              "paired_ci95_high_pp", "paired_ci_method", "mcnemar_two_sided_p", "timed_shots",
              "inference_seconds", "batch_size", "shots_per_second")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            labels = dict(mode=row["mode"], label=row["label"])
            timing = summary["timings"].get(row["mode"], {})
            writer.writerow(dict(scope="pooled", **labels, **rate_fields(row),
                                 timed_shots=timing.get("shots"), inference_seconds=timing.get("seconds"),
                                 batch_size=timing.get("batch_size"), shots_per_second=timing.get("shots_per_second")))
            for seed_row in row["per_seed"]:
                writer.writerow(dict(scope="seed", **labels, seed=seed_row["seed"], **rate_fields(seed_row)))
        for pair in summary["paired_comparisons"]:
            low, high = pair["paired_ci95_percentage_points"]
            labels = dict(mode=pair["candidate"], reference=pair["reference"], label=LABELS[pair["candidate"]])
            writer.writerow(dict(scope="paired", **labels, total=pair["total"], gains=pair["gains"], losses=pair["losses"],
                                 difference_percentage_points=pair["difference_percentage_points"], paired_ci95_low_pp=low,
                                 paired_ci95_high_pp=high, paired_ci_method=pair["paired_ci_method"],
                                 mcnemar_two_sided_p=pair["mcnemar_two_sided_p"]))
            for seed_row in pair["per_seed"]:
                writer.writerow(dict(scope="paired_seed", **labels, **seed_row))


def _plot_errors(path, rows):
    ink, secondary = "#172a38", "#526678"
    emphasis = {"mwpm": "#315a83", "balanced": "#00786d"}
    fig = Figure(figsize=(12, 7.6), facecolor="white")
    detail = fig.add_axes((0.255, 0.390, 0.525, 0.393))
    full = fig.add_axes((0.255, 0.115, 0.525, 0.117))
    positions = np.arange(len(rows))[::-1]
    rates = np.array([row["error_percent"] for row in rows])
    low, high = np.array([row["wilson95_percent"] for row in rows]).T
    colors = [emphasis.get(row["mode"], "#93a9bc") for row in rows]
    detail.barh(positions, rates, height=0.55, color=colors, zorder=2)
    detail.errorbar(rates, positions, xerr=np.maximum(0, [rates - low, high - rates]),
                    fmt="none", ecolor=ink, elinewidth=1.5, capsize=4, zorder=3)
    detail.set_xlim(0, min(100, high.max() * 1.17))
    detail.xaxis.set_major_locator(MaxNLocator(nbins=6))

    focus_rows = [row for row in rows if row["mode"] in emphasis]
    focus_positions = np.arange(len(focus_rows))[::-1]
    focus_rates = np.array([row["error_percent"] for row in focus_rows])
    full.barh(focus_positions, focus_rates, height=0.54, color=ink, label="Wrong logical bit", zorder=2)
    full.barh(focus_positions, 100 - focus_rates, left=focus_rates, height=0.54,
              color="#e2eaf0", label="Correct logical bit", zorder=2)
    full.set(xlim=(0, 100), xticks=np.arange(0, 101, 25))
    full.legend(loc="lower right", bbox_to_anchor=(1.0, 1.17), ncols=2, frameon=False,
                fontsize=9, labelcolor=secondary, borderaxespad=0, handlelength=1.6)

    for ax, panel_rows, y, heading_y in ((detail, rows, positions, 0.810),
                                         (full, focus_rows, focus_positions, 0.274)):
        ax.set(ylim=(-0.55, len(panel_rows) - 0.45), yticks=y,
               yticklabels=[row["label"] for row in panel_rows])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.tick_params(axis="both", length=0, labelsize=11, pad=8, colors=secondary)
        ax.grid(axis="x", color="#e1e8ef", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#bac9d6")

        for position, row, label in zip(y, panel_rows, ax.get_yticklabels()):
            highlighted = row["mode"] in emphasis
            color = emphasis.get(row["mode"], ink)
            weight = "bold" if highlighted else "normal"
            if highlighted:
                label.set_color(color)
                label.set_weight(weight)
            rate = row["error_percent"]
            rate_label = f"{rate:.4f}%" if rate == 0 or rate >= 0.0001 else f"{rate:.3g}%"
            ax.text(1.045, position, rate_label, transform=ax.get_yaxis_transform(), color=color,
                    weight=weight, va="center", fontsize=11)
            ax.text(1.205, position, f'{row["errors"]:,} / {row["total"]:,}', transform=ax.get_yaxis_transform(),
                    color=secondary, va="center", fontsize=10.5)
        fig.text(0.804, heading_y, "Rate", color=secondary, weight="bold", fontsize=10)
        fig.text(0.888, heading_y, "Wrong / total", color=secondary, weight="bold", fontsize=10)

    fig.text(0.035, 0.929, "Logical decoding error", color=ink, weight="bold", fontsize=24)
    fig.text(0.035, 0.883, f'Wrong final logical bit across {rows[0]["total"]:,} shared shots',
             color=secondary, fontsize=12)
    fig.text(0.255, 0.810, "Detail near zero", color=ink, weight="bold", fontsize=12)
    fig.text(0.255, 0.274, "Full 0–100% scale", color=ink, weight="bold", fontsize=12)
    fig.text(0.035, 0.039, "Whiskers in the detail view show 95% Wilson intervals for each method’s error rate.",
             color=secondary, fontsize=10)
    fig.savefig(path, dpi=160)


def report(out, truth, predictions, routed, seed_ids, timings):
    """Write the paired evaluation summary, timing data, and error-rate chart."""
    truth = _binary_vector(truth, "truth")
    seed_ids = np.asarray(seed_ids)
    if seed_ids.shape != truth.shape or seed_ids.dtype.kind not in "iu":
        raise ValueError("seed_ids must contain one integer per shot")
    if set(predictions) != set(LABELS):
        raise ValueError("predictions must cover the six report modes")

    routed = routed or {}
    no_queries = np.zeros_like(truth)
    groups = {int(seed): seed_ids == seed for seed in np.unique(seed_ids)}
    failures, rows = {}, []
    for mode, label in LABELS.items():
        failed = _binary_vector(predictions[mode], mode, truth.shape) != truth
        queried = _binary_vector(routed.get(mode, no_queries), f"{mode} routes", truth.shape)
        failures[mode] = failed
        per_seed = [dict(seed=seed, **_rates(failed[mask], queried[mask])) for seed, mask in groups.items()]
        rows.append(dict(mode=mode, label=label, **_rates(failed, queried), per_seed=per_seed))

    comparisons = [(mode, "mwpm") for mode in LABELS if mode != "mwpm"]
    comparisons.extend((("balanced", "margin_only"), ("learned", "raw_learned")))
    summary = {
        "rows": rows,
        "paired_comparisons": [_paired(mode, baseline, failures, groups) for mode, baseline in comparisons],
        "timings": _timing_rows(timings or {}),
        "metric_definitions": {
            "error_percent": "100 times wrong final logical bits divided by shots.",
            "wilson95_percent": "Marginal 95% Wilson score interval in percent.",
            "exact_route_percent": "100 times shots evaluated by the exact solver divided by shots.",
            "paired_comparisons": "Gains count reference errors fixed by the candidate. Losses count new errors. "
                                  "A negative percentage-point difference favors the candidate.",
            "mcnemar_two_sided_p": "Exact two-sided binomial test on discordant pairs, without a multiple-comparison adjustment.",
        },
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(out / "results.csv", summary)
    _plot_errors(out / "hybrid-errors.png", rows)
    return summary
