"""Report observed logical failures and conditional risks on shared shots."""

import csv
import json
from math import expm1, log
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.stats import binomtest

from bascd.stats import wilson95
from full_map_decoder import MAP_TIE_TOLERANCE
from hybrid_report import _binary_vector, _paired, _timing_rows


LABELS = {
    "mwpm": "PyMatching (MWPM)",
    "balanced": "Balanced FFT hybrid",
    "accuracy": "Exact on ambiguous shots",
    "full_map": "Full posterior MAP",
}


def _rate(failures, conditional_risk, conditional_excess):
    errors, total = int(failures.sum()), len(failures)
    return dict(
        errors=errors,
        total=total,
        error_percent=100 * errors / total,
        wilson95_percent=[100 * bound for bound in wilson95(errors, total)],
        expected_error_percent=100 * float(conditional_risk.mean()),
        expected_excess_vs_full_map_percentage_points=100 * float(conditional_excess.mean()),
    )


def _full_map_pair(reference, failures, groups):
    result = _paired("full_map", reference, failures, groups)
    gains, losses, shots = (result[key] for key in ("gains", "losses", "total"))
    lower, upper = result["paired_ci95_percentage_points"]
    if gains + losses == 0:
        # The absolute difference is bounded by the discordance probability.
        bound = -100 * expm1(log(0.05) / shots)
        result["paired_ci95_percentage_points"] = [-bound, bound]
        result["paired_ci_method"] = "Exact one-sided 95% Clopper-Pearson upper bound on discordance"
    elif lower == upper:
        # The normal approximation has no estimated variance when each seed
        # has a constant signed outcome. Use the sparse-pair method here too.
        gain_ci = binomtest(gains, shots).proportion_ci(confidence_level=0.975, method="exact")
        loss_ci = binomtest(losses, shots).proportion_ci(confidence_level=0.975, method="exact")
        result["paired_ci95_percentage_points"] = [100 * (loss_ci.low - gain_ci.high),
                                                   100 * (loss_ci.high - gain_ci.low)]
        result["paired_ci_method"] = "Conservative exact binomial bounds"
    else:
        result["paired_ci95_percentage_points"] = [max(-100, lower), min(100, upper)]
    return result


def summarize(truth, predictions, posterior, seed_ids, timings):
    """Summarize observed results and model-based risk for P(logical bit = 1)."""
    truth = _binary_vector(truth, "truth")
    seed_ids = np.asarray(seed_ids)
    if seed_ids.shape != truth.shape or seed_ids.dtype.kind not in "iu":
        raise ValueError("seed_ids must contain one integer per shot")
    if set(predictions) != set(LABELS):
        raise ValueError("predictions must cover all four report modes")
    posterior = np.asarray(posterior, dtype=np.float64)
    if posterior.shape != truth.shape or not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)):
        raise ValueError("posterior must contain one probability in [0, 1] per shot")

    decisions = {mode: _binary_vector(predictions[mode], mode, truth.shape) for mode in LABELS}
    full_map = decisions["full_map"]
    ties = np.abs(posterior - 0.5) <= MAP_TIE_TOLERANCE
    if np.any(full_map != np.where(ties, decisions["mwpm"], posterior > 0.5)):
        raise ValueError("full_map predictions must choose a posterior maximum or use MWPM on a numerical tie")
    failures = {mode: decision != truth for mode, decision in decisions.items()}
    groups = {int(seed): seed_ids == seed for seed in np.unique(seed_ids)}
    decision_margin = np.where(ties, 0, np.abs(2 * posterior - 1))
    rows = []
    for mode, label in LABELS.items():
        conditional_risk = np.where(decisions[mode], 1 - posterior, posterior)
        conditional_excess = decision_margin * (decisions[mode] != full_map)
        per_seed = [dict(seed=seed, **_rate(failures[mode][mask], conditional_risk[mask], conditional_excess[mask]))
                    for seed, mask in groups.items()]
        rows.append(dict(mode=mode, label=label, **_rate(failures[mode], conditional_risk, conditional_excess),
                         per_seed=per_seed))

    return {
        "rows": rows,
        "paired_comparisons": [_full_map_pair(mode, failures, groups) for mode in LABELS if mode != "full_map"],
        "timings": _timing_rows(timings or {}),
        "metric_definitions": {
            "error_percent": "100 times wrong final logical bits divided by shots.",
            "wilson95_percent": "Marginal 95% Wilson score interval in percent.",
            "expected_error_percent": "Mean conditional error probability of the decoder's choices under the exact "
                                      "posterior for the evaluated detector records, expressed as a percent.",
            "expected_excess_vs_full_map_percentage_points": "Mean absolute posterior margin |2p−1| across all shots, "
                                                            "expressed in percentage points. Agreement with full MAP "
                                                            "or a numerical tie contributes zero. The tie convention "
                                                            "changes the result by at most "
                                                            f"{200 * MAP_TIE_TOLERANCE:g} percentage points.",
            "map_tie": f"Use MWPM when the posterior is within {MAP_TIE_TOLERANCE:g} of 0.5.",
            "conditional_metrics": "Model-based calculations conditional on detector records. Observed error counts and "
                                   "paired intervals use the separately sampled logical truth.",
            "paired_comparisons": "Gains count baseline errors fixed by full MAP. Losses count new errors. "
                                  "A negative percentage-point difference favors full MAP.",
            "mcnemar_two_sided_p": "Exact two-sided binomial test on discordant pairs for each comparison.",
        },
    }


def _write_csv(path, summary):
    def rate_fields(row):
        low, high = row["wilson95_percent"]
        return {key: row[key] for key in rate_keys} | dict(wilson95_low_percent=low, wilson95_high_percent=high)

    rate_keys = ("errors", "total", "error_percent", "expected_error_percent",
                 "expected_excess_vs_full_map_percentage_points")
    pair_keys = ("total", "gains", "losses", "difference_percentage_points", "paired_ci_method",
                 "mcnemar_two_sided_p")
    fields = ("scope", "mode", "label", "seed", *rate_keys, "wilson95_low_percent", "wilson95_high_percent",
              "reference", "gains", "losses", "difference_percentage_points", "paired_ci95_low_pp",
              "paired_ci95_high_pp", "paired_ci_method", "mcnemar_two_sided_p", "timed_shots", "inference_seconds",
              "batch_size", "shots_per_second")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            labels = dict(mode=row["mode"], label=row["label"])
            timing = summary["timings"].get(row["mode"], {})
            writer.writerow(dict(scope="pooled", **labels, **rate_fields(row), timed_shots=timing.get("shots"),
                                 inference_seconds=timing.get("seconds"), batch_size=timing.get("batch_size"),
                                 shots_per_second=timing.get("shots_per_second")))
            for seed_row in row["per_seed"]:
                writer.writerow(dict(scope="seed", **labels, seed=seed_row["seed"], **rate_fields(seed_row)))
        for pair in summary["paired_comparisons"]:
            low, high = pair["paired_ci95_percentage_points"]
            labels = dict(mode=pair["candidate"], reference=pair["reference"], label=LABELS[pair["candidate"]])
            writer.writerow(dict(scope="paired", **labels, **{key: pair[key] for key in pair_keys},
                                 paired_ci95_low_pp=low, paired_ci95_high_pp=high))
            for seed_row in pair["per_seed"]:
                writer.writerow(dict(scope="paired_seed", **labels, **seed_row))


def _plot(path, rows):
    ink, secondary, teal = "#172a38", "#526678", "#00786d"
    fig = Figure(figsize=(10.6, 6.5), facecolor="white")
    detail = fig.add_axes((0.26, 0.515, 0.49, 0.26))
    full = fig.add_axes((0.26, 0.162, 0.49, 0.205))
    y = np.arange(len(rows))[::-1]
    rates = np.array([row["error_percent"] for row in rows])
    low, high = np.array([row["wilson95_percent"] for row in rows]).T
    colors = [teal if row["mode"] == "full_map" else "#557797" for row in rows]
    detail.barh(y, rates, height=0.55, color=colors, zorder=2)
    detail.errorbar(rates, y, xerr=np.maximum(0, [rates - low, high - rates]),
                    fmt="none", ecolor=ink, elinewidth=1.3, capsize=3.5, zorder=3)
    detail.set_xlim(0, min(100, high.max() * 1.16))
    detail.xaxis.set_major_locator(MaxNLocator(nbins=5))
    full.barh(y, rates, height=0.55, color=colors, zorder=2)
    full.barh(y, 100 - rates, left=rates, height=0.55, color="#e2eaf0", zorder=2)
    full.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100])

    for ax in (detail, full):
        ax.set(ylim=(-0.55, len(rows) - 0.45), yticks=y, yticklabels=[row["label"] for row in rows])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.tick_params(axis="both", length=0, labelsize=10, pad=7, colors=secondary)
        ax.grid(axis="x", color="#e1e8ef", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for name in ("top", "left", "right"):
            ax.spines[name].set_visible(False)
        ax.spines["bottom"].set_color("#bac9d6")
        ax.get_yticklabels()[-1].set(color=teal, weight="bold")

    decimals = max(4, len(str(rows[0]["total"])) - 2)
    for position, row in zip(y, rows, strict=True):
        label = f'{row["error_percent"]:.{decimals}f}%'
        row_color = teal if row["mode"] == "full_map" else ink
        detail.text(1.05, position, label, transform=detail.get_yaxis_transform(), color=row_color,
                    va="center", fontsize=10)
        detail.text(1.22, position, f'{row["errors"]:,} / {row["total"]:,}',
                    transform=detail.get_yaxis_transform(), color=secondary, va="center", fontsize=9.5)

    fig.text(0.042, 0.932, "Logical decoding error", color=ink, weight="bold", fontsize=23)
    fig.text(0.042, 0.886, f'Wrong final logical bits across {rows[0]["total"]:,} shared shots',
             color=secondary, fontsize=11)
    fig.text(0.26, 0.799, "Observed error rates", color=ink, weight="bold", fontsize=11)
    fig.text(0.775, 0.799, "Error rate", color=secondary, weight="bold", fontsize=9.5)
    fig.text(0.858, 0.799, "Wrong / total", color=secondary, weight="bold", fontsize=9.5)
    fig.text(0.26, 0.394, "Full 0–100% scale", color=ink, weight="bold", fontsize=11)
    fig.text(0.042, 0.055, "Whiskers show marginal 95% Wilson intervals. Paired differences are in the report tables.",
             color=secondary, fontsize=9.5)
    fig.savefig(path, dpi=160)


def write_report(out, summary):
    """Write JSON and CSV tables and a chart of observed error percentages."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    _write_csv(out / "results.csv", summary)
    _plot(out / "full-map-errors.png", summary["rows"])
    # The study uses results.json to recognize a completed report.
    pending = out / "results.json.tmp"
    pending.write_text(text, encoding="utf-8")
    pending.replace(out / "results.json")
