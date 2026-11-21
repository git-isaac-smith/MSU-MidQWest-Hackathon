"""Plot the fixed-noise transfer test beside the independent heralded-bath control."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bascd.stats import paired_ler_difference


def fixed_noise_points(transfer, replication):
    rows = json.loads(transfer.read_text(encoding="utf-8"))
    if replication is None:
        return rows[0]["shots"], {(r["prior"], r["decoder"]): r["paired_vs_native"]
                                  for r in rows if "paired_vs_native" in r}
    strata, seen = {}, set()
    manifest, shots = None, 0
    for summary in (transfer, replication):
        protocol = json.loads(summary.with_name("manifest.json").read_text(encoding="utf-8"))
        setup = [protocol[k] for k in ("noise", "scenarios", "shots_per_scenario", "profile_sha256", "bath_priors")]
        if manifest is not None and setup != manifest:
            raise ValueError("replications must use the same physical scenarios, shots, and priors")
        manifest = setup
        with summary.with_name("paired.csv").open(newline="", encoding="utf-8") as src:
            pairs = [r for r in csv.DictReader(src) if r["reference"] == "native"]
        digests = {r["sample_sha256"] for r in pairs}
        seeds = {r["seed"] for r in pairs}
        if len(digests) != len(protocol["scenarios"]) or seen & (digests | seeds):
            raise ValueError("pooled runs require independent samples")
        seen.update(digests | seeds)
        for r in pairs:
            strata.setdefault((r["prior"], r["decoder"]), []).append(
                (int(r["shots"]), int(r["recovered"]), int(r["lost"])))
        shots += len(digests) * protocol["shots_per_scenario"]
    return shots, {key: paired_ler_difference(*zip(*values)) for key, values in strata.items()}


def plot(transfer, research, output, replication=None):
    shots, contrasts = fixed_noise_points(transfer, replication)
    control = json.loads(research.read_text(encoding="utf-8"))
    colors = ("#186479", "#a64d25")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(left=.17, right=.98, top=.77, bottom=.21, wspace=.55)
    fig.suptitle("Surface-code decoding with a structured-bath calibration",
                 fontsize=19, weight="bold", y=.97)

    left.set_title("Fixed-noise transfer", loc="left", fontsize=15, weight="bold", pad=28)
    left.text(0, 1.04, f'{shots:,} paired shots per method · provisional Stim circuits',
              transform=left.transAxes, fontsize=9)
    left.axvline(0, color="#828887", lw=1.1)
    priors = ("expanded_uniform", "clock", "selected")
    for i, (kind, label) in enumerate((("mwpm", "Ordinary"), ("correlated_mwpm", "Correlated"))):
        for j, prior in enumerate(priors):
            paired = contrasts[prior, kind]
            value = paired["ler_difference"]
            interval = paired["approximate_95_ci"]
            err = ([[100 * (value - interval[0])], [100 * (interval[1] - value)]] if interval else None)
            left.errorbar(100 * value, j + (i - .5) * .25, xerr=err, color=colors[i],
                          fmt=("o", "s")[i], capsize=3, label=label if j == 0 else None)
    left.set_yticks(range(3), ["Uniform, expanded", "Bath, fixed clock", "Bath, selected time"])
    left.set_ylim(2.5, -.5)
    left.set_xlabel("Change in logical failure rate\nvs. native weights (percentage points)")
    left.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False)

    right.set_title("Location-heralded bath simulation", loc="left", fontsize=15, weight="bold", pad=28)
    fixed = control["clocks"]["fixed"]
    right.text(0, 1.04, f'{fixed["shots"]:,} paired shots per time · distance {control["distance"]} · ideal Z memory',
               transform=right.transAxes, fontsize=9)
    for i, (key, label) in enumerate((("blind", "Syndrome only"), ("heralded", "Syndrome + flags"))):
        for j, clock in enumerate(("fixed", "selected")):
            row = control["clocks"][clock][key]
            rate = 100 * row["logical_failure_fraction"]
            low, high = (100 * v for v in row["wilson_95"])
            x = j + (i - .5) * .36
            right.bar(x, rate, .33, color=colors[i], label=label if j == 0 else None)
            right.errorbar(x, rate, yerr=[[rate - low], [high - rate]], color="#232a2b", capsize=3)
            right.text(x, high + 1, f"{rate:.2f}%", ha="center", fontsize=9)
    right.set_xticks([0, 1], [f'Fixed clock\nt = {fixed["time"]:g}',
                              f'Selected time\nt = {control["clocks"]["selected"]["time"]:g}'])
    right.set_ylim(0, max(100 * group[k]["wilson_95"][1] for group in control["clocks"].values()
                          for k in ("blind", "heralded")) * 1.17)
    right.set_ylabel("Logical failure rate (%)")
    right.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
    left.grid(axis="x", alpha=.15)
    right.grid(axis="y", alpha=.15)
    right.set_axisbelow(True)
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=150, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transfer", type=Path, help="bath_transfer summary.json")
    parser.add_argument("research", type=Path, help="bath_erasure result.json")
    parser.add_argument("output", type=Path, help="new PNG path")
    parser.add_argument("--replication", type=Path, help="independent second run's summary.json to pool")
    args = parser.parse_args()
    plot(args.transfer, args.research, args.output, args.replication)
