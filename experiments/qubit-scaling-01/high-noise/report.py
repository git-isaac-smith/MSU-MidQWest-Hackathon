"""Render the verified higher-noise stress results as a standalone HTML table."""

import argparse
import csv
from html import escape
from pathlib import Path


LABELS = {
    "mwpm": "Ordinary MWPM",
    "correlated": "Native correlated MWPM",
    "expanded_corr": "Equal-axis correlated MWPM",
    "clock_corr": "Bath-clock correlated MWPM",
    "union_find": "Library weighted Union-Find",
    "css_routed": "CSS activity routing",
}


def render(source, output):
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lines = []
    last_distance = None
    for row in rows:
        distance = row["distance"]
        start_group = distance != last_distance
        last_distance = distance
        label = LABELS[row["method"]]
        if row["status"] == "ok":
            accuracy = f'{float(row["ler_pct"]):.3f}% <small>{row["errors"]}/{row["shots"]}</small>'
            interval = f'{float(row["wilson_low_pct"]):.2f}–{float(row["wilson_high_pct"]):.2f}%'
            timing = f'{float(row["batch_us_per_shot"]):,.2f}'
            paired = []
            for reference in ("mwpm", "expanded_corr"):
                if row.get(f"{reference}_recovered") and row["method"] != reference:
                    paired.append(f'{row[f"{reference}_recovered"]} / {row[f"{reference}_regressed"]}')
                else:
                    paired.append("—")
            cells = f"<td>{accuracy}</td><td>{interval}</td><td>{timing}</td>" + "".join(
                f"<td>{item}</td>" for item in paired
            )
        else:
            cells = f'<td colspan="5">{escape(row["reason"])}</td>'
        lines.append(f'<tr class="{"section" if start_group else ""}"><td>d{distance}</td>'
                     f'<th scope="row">{label}</th>{cells}</tr>')
    largest = {row["method"]: row for row in rows if row["distance"] == "31" and row["status"] == "ok"}
    headline = ""
    if {"mwpm", "clock_corr", "expanded_corr"} <= largest.keys():
        ordinary, bath, control = (largest[name] for name in ("mwpm", "clock_corr", "expanded_corr"))
        headline = (
            f'<p>At distance 31, ordinary matching failed on {ordinary["errors"]} of {ordinary["shots"]} '
            f'shots. The equal-axis and bath-clock correlated variants failed on {control["errors"]} and '
            f'{bath["errors"]} shots, respectively. The bath-clock versus equal-axis comparison had '
            f'{bath["expanded_corr_recovered"]} recoveries and {bath["expanded_corr_regressed"]} regressions.</p>'
        )
    output.write_text("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Higher-noise decoder stress test</title>
<style>
:root { color-scheme: light; --ink: #182628; --muted: #4a5f62; --line: #dce6e3; --accent: #00624f; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f8f6; color: var(--ink); font: 1rem/1.55 system-ui,sans-serif; }
main { max-width: 1150px; margin: auto; padding: clamp(1rem,5vw,4rem); }
h1 { font-size: clamp(2rem,5vw,3.4rem); line-height: 1.1; letter-spacing: -.035em; margin: .25em 0; }
h2 { font-size: 1.35rem; margin: 2rem 0 .35rem; }
p { max-width: 82ch; margin: .65rem 0; }
a { color: var(--accent); text-underline-offset: .18em; }
.data { overflow: auto; border: 1px solid var(--line); border-radius: .7rem; margin-top: 1.5rem; background: white; }
table { border-collapse: collapse; min-width: 53rem; width: 100%; font-variant-numeric: tabular-nums; }
caption { text-align: left; padding: .9rem 1rem; color: var(--muted); }
thead { background: #e6f2ed; }
th,td { text-align: right; padding: .55rem .75rem; border-top: 1px solid var(--line); }
th:nth-child(2), td:first-child { text-align: left; }
td small { color: var(--muted); white-space: nowrap; }
tr.section td, tr.section th { border-top: 2px solid #86ad9f; }
@media print { main { max-width: none; padding: 0; } .data { overflow: visible; } table { min-width: 0; } th,td { padding: .3rem; } }
</style>
</head>
<body><main>
<h1>Higher-noise decoder stress test</h1>
<p>Rotated Z memory at distances 7, 15 and 31, with rounds equal to distance and a fixed parameter of 1% in the provisional circuit-level noise model. Six decoders receive the same 2,048 detector records at each distance. The observable stays with the evaluator.</p>
""" + headline + """
<div class="data" role="region" aria-label="Accuracy and runtime by code distance" tabindex="0">
<table>
<caption>Logical error, warm batch time, and paired recoveries / regressions against two references.</caption>
<thead><tr><th scope="col">Code</th><th scope="col">Method</th><th scope="col">Logical error</th><th scope="col">95% Wilson</th><th scope="col">µs/shot</th><th scope="col">vs ordinary</th><th scope="col">vs equal-axis</th></tr></thead><tbody>
""" + "\n".join(lines) + """
</tbody></table></div>
<h2>Protocol</h2>
<p>Equal-axis and bath-clock matching both compile correlated matchers using the same idle-channel expansion, with uniform or archived bath-derived Pauli weights. CSS routing sends records above the activity cutoff to native correlated matching. The cutoff uses 1,024 separate unlabeled calibration shots and a 6% target fraction.</p>
<p>Runtime is the median from warm 128-shot calls in separate single-process decoder workers. Runtime includes any required input unpacking. Setup, sampling and scoring occur outside the timer. The main benchmark was also active on the host.</p>
<p><a href="README.md">Methods and replay</a> · <a href="measured/metrics.csv">All results and paired intervals</a> · <a href="measured/protocol.json">Frozen protocol</a></p>
</main></body></html>
""", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render(args.metrics, args.out)

