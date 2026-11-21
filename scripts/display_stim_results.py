"""Display a Stim shot archive as a median/best decoder performance table."""

import argparse
import csv
from pathlib import Path
from statistics import median


DECODERS = {
    "native": "Standard correlated matching",
    "bath_clock": "Bath-weighted correlated matching",
    "frozen": "Distance-based hybrid decoder",
}
MEMORIES = {"x": "X memory · phase flips", "z": "Z memory · bit flips"}


def performance_table(archive):
    with (archive / "stats.csv").open(encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))
    lines = [
        "| Decoder | Median logical success | Best observed logical success | Median batch time (µs/shot) |",
        "|---|---:|---:|---:|",
    ]
    cases = {r["case"] for r in rows}
    for name, label in DECODERS.items():
        group = [r for r in rows if r["decoder"] == name]
        if not group or {r["case"] for r in group} != cases or len(group) != len(cases):
            raise ValueError(f"expected one {name} measurement in each scenario")
        success = [1 - int(r["failures"]) / int(r["shots"]) for r in group]
        time_us = median(float(r["batch_us_per_shot"]) for r in group)
        lines.append(f"| {label} | {median(success):.2%} | {max(success):.2%} | {time_us:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="directory containing stats.csv")
    args = parser.parse_args()
    print(performance_table(args.archive))
