"""Render an offline HTML report from a geometric decoder scaling run."""

import argparse
import csv
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path


COLORS = ("#233850", "#006f84", "#aa4f13", "#6951a8", "#177158", "#a5355e")
DASHES = ("", "7 4", "3 3", "10 3 2 3", "2 5", "6 2 2 2")
METHOD_LABELS = {
    "mwpm": "Ordinary MWPM",
    "ordinary": "Ordinary MWPM",
    "correlated": "Correlated MWPM",
    "belief": "Belief matching",
    "activity": "Activity routing",
    "learned": "Learned routing",
    "gap2": "Logical-gap routing",
}

STYLE = """
:root {
  color-scheme: light; color: #182638; background: #eef3f5;
  font-family: system-ui, sans-serif; font-synthesis: none;
}
* { box-sizing: border-box }
[hidden] { display: none !important }
body { margin: 0 }
header { background: #14263a; color: #f6fafc; border-top: .35rem solid #49c5bf }
header > div, main { width: min(96%, 82rem); margin: auto; padding: 2.15rem 1rem }
header h1 { font-size: clamp(2rem, 4.5vw, 3.3rem); letter-spacing: -.05em; margin: 0 }
header p { max-width: 51rem; margin: .6rem 0 0; color: #c7d9e6 }
h2 { letter-spacing: -.025em; margin: 0 0 .6rem }
a { color: #006581; text-underline-offset: .2em }
hr { border: 0; border-top: 1px solid #d9e4e8; margin: 1.3rem 0 }
.summary { display: flex; flex-wrap: wrap; gap: .65rem 1.75rem; margin: 1.25rem 0 0 }
.summary div { min-width: 7rem }
.summary dt { color: #c7d9e6; font-size: .85rem }
.summary dd { margin: .1rem 0 0; font-size: 1.35rem; font-weight: 650 }
.panel { background: #fff; border: 1px solid #dce6eb; border-radius: .9rem;
  margin: 0 0 1rem; padding: clamp(.9rem, 2vw, 1.5rem); min-width: 0 }
.charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem }
.charts .panel { margin: 0 }
.condition { margin-top: 1.35rem }
.condition > .panel { margin-top: 1rem }
.condition > h2 { font-size: 1.2rem }
.controls { display: flex; flex-wrap: wrap; gap: .65rem 1.2rem; margin: 0 0 1.5rem }
.controls label { font-weight: 650; margin-right: .6rem }
select { font: inherit; color: inherit; background: #fff; padding: .45rem .7rem;
  border: 1px solid #9fb4bd; border-radius: .4rem; max-width: 100% }
figure { margin: 0 }
figure h3 { margin: 0; font-size: 1.08rem }
figure p, .note { color: #4c6274; margin: .3rem 0 .7rem; font-size: .91rem }
.chart { display: block; width: 100%; height: auto }
.legend { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: .4rem 1rem;
  margin: .8rem 0; font-size: .88rem }
.legend li { display: flex; align-items: center; gap: .35rem }
.snapshot { margin: .75rem 0; padding: .75rem 1rem; border-left: 3px solid #007b85;
  background: #e1eff1; border-radius: 0 .4rem .4rem 0 }
.snapshot p { margin: .22rem 0 }
.related { border-left: .28rem solid #007b85 }
.related h2 { font-size: 1.12rem }
.related p { margin: .25rem 0 0 }
.pilot-key { color: #4c6274 }
.scroll { overflow: auto }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums }
caption { text-align: left; font-weight: 700; margin: 0 0 .55rem }
thead { color: #435a6c; background: #f0f5f7 }
th, td { padding: .58rem .62rem; border-bottom: 1px solid #e1e9ed;
  text-align: right; white-space: nowrap; font-size: .82rem }
th:first-child, td:first-child, .coverage th, .coverage td { text-align: left }
.coverage th, .coverage td { white-space: normal; vertical-align: top }
.coverage th:first-child { width: 25% }
tr.group-start td { border-top: 2px solid #cbdbe2 }
details > summary { cursor: pointer; font-weight: 650 }
.row-reason { min-width: 11rem; max-width: 22rem; margin: .3rem 0 0 auto;
  text-align: left; white-space: normal; font-weight: 400 }
.protocol { display: grid; grid-template-columns: minmax(7rem, 13rem) minmax(0, 1fr); gap: .5rem 1rem }
.protocol dt { font-weight: 650 }
.protocol dd { margin: 0; overflow-wrap: anywhere }
.links { display: flex; flex-wrap: wrap; gap: .65rem 1rem; margin: .75rem 0 0 }
@media (max-width: 52rem) { .charts { grid-template-columns: 1fr } }
@media (max-width: 36rem) { .protocol { display: block } .protocol dd { margin: .25rem 0 .75rem } }
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _label(method: str, labels: dict[str, str]) -> str:
    return labels.get(method, method.replace("_", " ").replace("-", " ").capitalize())


def _display(row: dict[str, str], key: str, digits: int | None = None) -> str:
    value = _number(row, key)
    if value is None:
        return "—"
    if digits is None:
        return f"{int(value):,}"
    if value and abs(value) < 10 ** -digits:
        return f"{value:.2g}"
    return f"{value:,.{digits}f}"


def _numeric_axis(values: list[float], metric: str) -> tuple[float, float, list[float], bool]:
    largest = max(values)
    smallest = min(values)
    log_axis = metric == "batch_us_per_shot" and smallest > 0 and largest / smallest >= 10
    if log_axis:
        low_power = math.floor(math.log10(smallest))
        high_power = math.ceil(math.log10(largest))
        ticks = [10.0 ** i for i in range(low_power, high_power + 1)]
        return ticks[0], ticks[-1], ticks, True
    if largest <= 0:
        largest = 1.0
    unit = 10 ** math.floor(math.log10(largest / 4))
    step = next(unit * factor for factor in (1, 2, 2.5, 5, 10) if unit * factor >= largest / 4)
    count = math.ceil(largest / step)
    ticks = [step * index for index in range(count + 1)]
    return 0.0, ticks[-1], ticks, False


def _chart(rows, sizes, styles, labels, metric, heading, description):
    by_method = defaultdict(dict)
    values = []
    show_intervals = metric == "ler_pct"
    for row in rows:
        value = _number(row, metric)
        if value is None:
            continue
        by_method[row["method"]][int(row["physical_qubits"])] = (value, row)
        values.append(value)
        if show_intervals:
            for key in ("ci_low_pct", "ci_high_pct"):
                bound = _number(row, key)
                if bound is not None:
                    values.append(bound)
    result = [
        f'<figure class="panel"><h3>{heading}</h3>',
        f'<p>{description}</p>',
    ]
    if not values:
        result.append("<p>No measurements for this setting.</p></figure>")
        return "".join(result)
    low, high, ticks, logarithmic = _numeric_axis(values, metric)
    width, height = 610, 320
    left, right, top, bottom = 59, width - 34, 25, height - 69
    first_qubits, last_qubits = min(sizes), max(sizes)
    log_x = last_qubits / first_qubits >= 10
    scale_x = math.log10 if log_x else float
    left_x = scale_x(first_qubits)
    x_step = (right - left) / (scale_x(last_qubits) - left_x) if len(sizes) > 1 else 0
    lower_y = math.log10(low) if logarithmic else low
    upper_y = math.log10(high) if logarithmic else high

    def x(qubits):
        if len(sizes) == 1:
            return (left + right) / 2
        return left + (scale_x(qubits) - left_x) * x_step

    def y(value):
        scaled = math.log10(value) if logarithmic else value
        return bottom - (scaled - lower_y) / (upper_y - lower_y) * (bottom - top)

    y_label = "%" if show_intervals else "µs / shot"
    x_label = "Physical qubits · log scale" if log_x else "Physical qubits"
    axis_note = " · log scale" if logarithmic else ""
    result.append(
        f'<svg class="chart" role="img" aria-label="{heading} by physical qubits. '
        f'Exact values are available in the table." viewBox="0 0 {width} {height}" '
        'font-family="system-ui, sans-serif" font-size="13">'
        f'<text x="{left}" y="14" fill="#4c6274">{y_label}{axis_note}</text>'
    )
    for tick in ticks:
        point_y = y(tick)
        result.append(
            f'<path d="M {left} {point_y:.2f} H {right}" stroke="#dfe9ee"/>'
            f'<text x="{left - 8}" y="{point_y + 4:.2f}" text-anchor="end" fill="#4c6274">{tick:.4g}</text>'
        )
    for qubits, distance in sizes.items():
        point_x = x(qubits)
        result.append(
            f'<path d="M {point_x:.2f} {bottom} v 5" stroke="#8ea4af"/>'
            f'<text x="{point_x:.2f}" y="{bottom + 21}" text-anchor="middle" fill="#30485c">{qubits:,}</text>'
            f'<text x="{point_x:.2f}" y="{bottom + 38}" text-anchor="middle" fill="#4c6274">d={distance}</text>'
        )
    result.append(
        f'<text x="{(left + right) / 2}" y="{height - 7}" text-anchor="middle" '
        f'fill="#30485c">{x_label}</text>'
    )
    for method, points in by_method.items():
        color, dash = styles[method]
        result.append("<g>")
        path = []
        previous_group = None
        for qubits in sizes:
            if qubits not in points:
                previous_group = None
                continue
            value, row = points[qubits]
            status = row.get("status") or "ok"
            group = (status, row.get("batch_size") if not show_intervals else None)
            command = "L" if group == previous_group else "M"
            path.append(f"{command} {x(qubits):.2f} {y(value):.2f}")
            previous_group = group
        result.append(
            f'<path d="{" ".join(path)}" stroke="{color}" stroke-width="2" '
            f'stroke-dasharray="{dash}" fill="none"/>'
        )
        for qubits, (value, row) in points.items():
            px, py = x(qubits), y(value)
            if show_intervals:
                ci_low = _number(row, "ci_low_pct")
                ci_high = _number(row, "ci_high_pct")
                if ci_low is not None and ci_high is not None:
                    result.append(
                        f'<path d="M {px:.2f} {y(ci_low):.2f} V {y(ci_high):.2f} '
                        f'M {px - 3:.2f} {y(ci_low):.2f} h 6 M {px - 3:.2f} {y(ci_high):.2f} h 6" '
                        f'stroke="{color}" stroke-width="1.3" opacity=".72"/>'
                    )
            status = row.get("status") or "ok"
            description = f"{_label(method, labels)} · d={row['distance']} · {value:.5g} {y_label}"
            if status != "ok":
                description += f" · {status} ({_display(row, 'shots')} accuracy shots)"
            title = escape(description)
            if status == "pilot":
                result.append(
                    f'<path d="M {px:.2f} {py - 5:.2f} l 5 5 -5 5 -5 -5 Z" fill="#fff" '
                    f'stroke="{color}" stroke-width="1.8"><title>{title}</title></path>'
                )
            else:
                fill = color if status == "ok" else "#fff"
                result.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" fill="{fill}" '
                    f'stroke="{color}" stroke-width="1.8"><title>{title}</title></circle>'
                )
        result.append("</g>")
    result.append("</svg></figure>")
    return "".join(result)


def _legend(methods, styles, labels, has_pilot) -> str:
    entries = []
    for method in methods:
        color, dash = styles[method]
        entries.append(
            '<li><svg width="28" height="13" aria-hidden="true">'
            f'<path d="M 0 6.5 H 28" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
            f'<circle cx="14" cy="6.5" r="3" fill="{color}"/></svg>'
            f'{escape(_label(method, labels))}</li>'
        )
    if has_pilot:
        entries.append(
            '<li class="pilot-key"><svg width="14" height="13" aria-hidden="true">'
            '<path d="M 7 2.5 l 4 4 -4 4 -4 -4 Z" fill="#fff" stroke="#4c6274" '
            'stroke-width="1.5"/></svg>Pilot</li>'
        )
    return '<ul class="legend" aria-label="Decoder legend">' + "".join(entries) + "</ul>"


def _figures(rows, sizes, methods, styles, labels) -> str:
    visible = {row["method"] for row in rows}
    legend = _legend(
        [method for method in methods if method in visible],
        styles, labels, any(row.get("status") == "pilot" for row in rows),
    )
    return "".join([
        legend,
        '<div class="charts">',
        _chart(rows, sizes, styles, labels, "ler_pct", "Logical error rate",
               "Percentage of shots with a logical failure. Whiskers show 95% Wilson intervals."),
        _chart(rows, sizes, styles, labels, "batch_us_per_shot", "Decoder runtime",
               "Median wall time per shot in warmed, sequential, full-size batch calls."),
        "</div>",
    ])


def _table(rows, labels) -> str:
    headers = (
        "Decoder", "Distance", "Data qubits", "Physical qubits", "Rounds",
        "Failures / shots", "LER %", "95% CI %", "Batch µs/shot",
        "Setup s", "Batch shots", "Recovered / lost", "Status",
    )
    result = [
        '<div class="scroll" role="region" tabindex="0" aria-label="Measured values table">'
        '<table><caption>Measured values</caption><thead><tr>',
        *(f'<th scope="col">{header}</th>' for header in headers),
        "</tr></thead><tbody>",
    ]
    previous_distance = None
    for row in rows:
        distance = int(row["distance"])
        group_start = previous_distance is not None and distance != previous_distance
        previous_distance = distance
        cells = [
            _label(row["method"], labels),
            *(_display(row, key) for key in ("distance", "data_qubits", "physical_qubits", "rounds")),
            f'{_display(row, "failures")} / {_display(row, "shots")}',
            _display(row, "ler_pct", 4),
            f'{_display(row, "ci_low_pct", 4)}–{_display(row, "ci_high_pct", 4)}'
            if _number(row, "ci_low_pct") is not None and _number(row, "ci_high_pct") is not None else "—",
            _display(row, "batch_us_per_shot", 3),
            _display(row, "setup_s", 3),
            _display(row, "batch_size"),
            f'{_display(row, "recovered")} / {_display(row, "lost")}'
            if _number(row, "recovered") is not None or _number(row, "lost") is not None else "—",
        ]
        group = ' class="group-start"' if group_start else ""
        result.append(f'<tr{group} data-method="{escape(row["method"])}">')
        result.extend(f"<td>{escape(cell)}</td>" for cell in cells)
        status = escape(row.get("status") or "ok")
        reason = escape(row.get("reason") or "")
        target = _display(row, "planned_shots")
        if target != "—" and _number(row, "shots") != _number(row, "planned_shots"):
            status += f" · {target} planned"
        if reason:
            result.append(
                f'<td><details><summary>{status}</summary><p class="row-reason">{reason}</p></details></td>'
            )
        else:
            result.append(f"<td>{status}</td>")
        result.append("</tr>")
    result.append("</tbody></table></div>")
    return "".join(result)


def _snapshot(rows, labels, baseline_method) -> str:
    completed = [
        row for row in rows
        if (row.get("status") or "ok") == "ok" and _number(row, "shots") and _number(row, "ler_pct") is not None
    ]
    references = [row for row in completed if row["method"] == baseline_method]
    if not references:
        return ""
    baseline = max(references, key=lambda row: int(row["distance"]))
    distance = int(baseline["distance"])
    shots = _number(baseline, "shots")
    comparable = [
        row for row in completed
        if int(row["distance"]) == distance and _number(row, "shots") == shots
    ]
    count = len(comparable)
    if count == 1:
        comparison = "Ordinary MWPM was the only decoder with this completed shot count at this size."
    else:
        comparison = f"{count} decoders completed the same {int(shots):,}-shot accuracy sample at this size."
    return (
        '<div class="snapshot">'
        f'<p><strong>Largest completed ordinary MWPM run</strong> · d={distance} · '
        f'{_display(baseline, "physical_qubits")} physical qubits</p>'
        f'<p>{escape(_label(baseline["method"], labels))} recorded '
        f'{_display(baseline, "failures")} failures in {int(shots):,} shots '
        f'({_display(baseline, "ler_pct", 4)}%). {comparison}</p></div>'
    )


def _coverage(rows, labels) -> str:
    result = ['<section class="panel" aria-label="Benchmark coverage"><h2>Coverage and limits</h2>']
    if rows:
        result.append(
            '<div class="scroll"><table class="coverage"><thead><tr>'
            '<th scope="col">Decoder</th><th scope="col">Scope</th><th scope="col">Reason</th>'
            '</tr></thead><tbody>'
        )
        for row in rows:
            method = row.get("method", "")
            values = [_label(method, labels), row.get("scope", ""), row.get("reason", "")]
            result.append("<tr>" + "".join(f"<td>{escape(value or '—')}</td>" for value in values) + "</tr>")
        result.append("</tbody></table></div>")
    else:
        result.append("<p>No coverage limits were recorded in this run.</p>")
    result.append("</section>")
    return "".join(result)


def _protocol(protocol) -> str:
    result = ['<section class="panel"><details><summary>Run protocol</summary><dl class="protocol">']
    for key, value in protocol.items():
        if key == "method_labels":
            continue
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        title = escape(key.replace("_", " ").capitalize())
        result.append(f"<dt>{title}</dt><dd>{escape(value)}</dd>")
    result.append("</dl></details></section>")
    return "".join(result)


def _related_results(run: Path) -> str:
    related = []
    if (run / "high-noise" / "index.html").is_file():
        related.append(
            '<p><a href="high-noise/index.html"><strong>Decoder comparison at 1% physical noise</strong></a> '
            'on fresh Z-memory records through distance 31.</p>'
        )
    if (run / "heralds" / "index.html").is_file():
        related.append(
            '<p><a href="heralds/index.html"><strong>Bath erasure scaling results</strong></a> '
            'for the fixed and bath-selected check times, with the location flags hidden and visible.</p>'
        )
    if (run / "fft" / "index.html").is_file():
        related.append(
            '<p><a href="fft/index.html"><strong>FFT and CNN scaling results</strong></a> '
            'from a separate five-round Z-memory model measured through 1,089 data qubits.</p>'
        )
    if (run / "report.md").is_file():
        related.append('<p><a href="report.md">Written comparison and interpretation</a></p>')
    if (run / "d31-paired.csv").is_file():
        related.append(
            '<p><a href="d31-paired.csv">Distance-31 paired differences and confidence intervals (CSV)</a></p>'
        )
    if (run.parent / "bath-confirmation-01" / "README.md").is_file():
        related.append(
            '<p><a href="../bath-confirmation-01/README.md">Archived flag-only demo</a> '
            'from the bath confirmation study.</p>'
        )
    if not related:
        return ""
    return '<section class="panel related"><h2>Related results</h2>' + "".join(related) + "</section>"


def _large_case_table(rows, labels):
    """Show directly comparable full-sample errors and timings on first open."""
    largest = max(int(row["distance"]) for row in rows)
    primary = ("mwpm", "correlated", "expanded_corr", "clock_corr")
    selected = {(row["basis"].lower(), row["method"]): row for row in rows
                if int(row["distance"]) == largest and (row.get("status") or "ok") == "ok" and row["method"] in primary}
    if not all((basis, method) in selected for basis in ("x", "z") for method in primary):
        return ""
    case = selected["z", "mwpm"]
    shots = int(case["shots"])
    if any(int(row["shots"]) != shots or row["p"] != case["p"] for row in selected.values()):
        return ""
    parts = [f'<section class="panel"><h2>Distance {largest} · {int(case["physical_qubits"]):,} physical qubits</h2>',
             f'<p class="note">Each method decoded the same {shots:,} shots per basis at '
             f'{float(case["p"])*100:g}% physical noise. Times are batch microseconds per shot.</p>',
             '<div class="scroll"><table><thead><tr><th>Decoder</th><th>X memory error</th><th>X time</th>'
             '<th>Z memory error</th><th>Z time</th></tr></thead><tbody>']
    for method in primary:
        parts.append(f'<tr><td>{escape(_label(method, labels))}</td>')
        for basis in ("x", "z"):
            row = selected[basis, method]
            parts.append(f'<td>{float(row["ler_pct"]):.2f}% ({int(row["failures"]):,})</td>'
                         f'<td>{float(row["batch_us_per_shot"]):,.1f} µs</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div><p class="note">Parentheses give observed failures. '
                 'See the charts below for 95% intervals, pilot sizes and all decoder configurations.</p></section>')
    return "".join(parts)


def build_report(run: Path) -> Path:
    """Write the completed study's measurements and coverage to run/index.html."""
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8-sig"))
    rows = _read_csv(run / "metrics.csv")
    coverage = _read_csv(run / "coverage.csv")
    if not rows:
        raise ValueError(f"No measurement rows in {run / 'metrics.csv'}")
    labels = METHOD_LABELS | protocol.get("method_labels", {})
    present_methods = {row["method"] for row in rows}
    baseline_method = next((key for key in ("mwpm", "ordinary") if key in present_methods), "")
    default_comparison = "correlated" if baseline_method and "correlated" in present_methods else ""
    methods = sorted(present_methods, key=lambda name: (name != baseline_method, name))
    measured_methods = {
        row["method"] for row in rows
        if _number(row, "ler_pct") is not None or _number(row, "batch_us_per_shot") is not None
    }
    styles = {
        method: (COLORS[index % len(COLORS)], DASHES[(index + index // len(COLORS)) % len(DASHES)])
        for index, method in enumerate(methods)
    }
    distances = sorted({int(row["distance"]) for row in rows})
    physical_qubits = sorted({int(row["physical_qubits"]) for row in rows})
    settings = defaultdict(list)
    for row in rows:
        settings[row["basis"].upper(), float(row["p"])].append(row)

    title = "Logical error and decoder runtime"
    subtitle = "Rotated surface-code memory benchmarks at increasing qubit counts."
    qubit_range = f"{physical_qubits[0]:,}"
    if len(physical_qubits) > 1:
        qubit_range += f"–{physical_qubits[-1]:,}"
    result = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title}</title><style>{STYLE}</style></head><body>',
        f"<header><div><h1>{title}</h1><p>{subtitle}</p>",
        '<dl class="summary">',
        f'<div><dt>Physical qubits</dt><dd>{qubit_range}</dd></div>',
        f'<div><dt>Code distances</dt><dd>{", ".join(map(str, distances))}</dd></div>',
        f'<div><dt>Measured decoders</dt><dd>{len(measured_methods)}</dd></div>',
        '</dl></div></header><main>',
        _large_case_table(rows, labels),
        _related_results(run),
    ]
    result.append(f'<div class="controls" id="controls" data-baseline="{escape(baseline_method)}" hidden>')
    if len(settings) > 1:
        result.append('<div><label for="setting">Benchmark setting</label><select id="setting">')
        for index, (basis, p) in enumerate(sorted(settings)):
            result.append(f'<option value="{index}">{escape(basis)} memory · p = {p:.6g}</option>')
        result.append("</select></div>")
    if len(methods) > 1:
        result.append('<div><label for="decoder">Decoder comparison</label><select id="decoder">')
        result.append('<option value="">All decoders</option>')
        for method in methods:
            label = escape(_label(method, labels))
            if method != baseline_method and baseline_method:
                label += " vs ordinary MWPM"
            selected = " selected" if method == default_comparison else ""
            result.append(f'<option value="{escape(method)}"{selected}>{label}</option>')
        result.append("</select></div>")
    result.append("</div>")
    for index, ((basis, p), observations) in enumerate(sorted(settings.items())):
        observations.sort(key=lambda row: (int(row["distance"]), methods.index(row["method"])))
        sizes = dict(sorted({int(row["physical_qubits"]): int(row["distance"]) for row in observations}.items()))
        result.extend([
            f'<section class="condition" data-setting="{index}"><h2>{escape(basis)} memory · p = {p:.6g}</h2>',
            _snapshot(observations, labels, baseline_method),
            '<div data-chart-view data-selection="">',
            _figures(observations, sizes, methods, styles, labels),
            "</div>",
        ])
        if len(methods) > 1:
            for method in methods:
                comparison = [row for row in observations if row["method"] in (baseline_method, method)]
                result.extend([
                    f'<template data-chart-method="{escape(method)}">',
                    _figures(comparison, sizes, methods, styles, labels),
                    "</template>",
                ])
        result.extend([
            '<div class="panel">',
            _table(observations, labels),
            '<p class="note">Physical qubits include active data and measurement ancillas. '
            'Recovered and lost shots are paired changes versus ordinary MWPM on the row’s accuracy sample. '
            'Open diamonds mark pilots on shared prefixes of 16 to 4,096 shots. The status column identifies follow-ups after time limits. '
            'Open circles mark other partial measurements.</p>'
            '<p class="note">Timing starts with unpacked detector records and includes internal packing. '
            'The inference timer excludes sampling, disk I/O, scoring, and setup. '
            'Batches contain 128 shots. Pilots with only 16 accuracy shots use 16-shot batches.</p></div></section>',
        ])
    result.extend([
        _coverage(coverage, labels),
        _protocol(protocol),
        '<nav class="links" aria-label="Run files"><a href="metrics.csv">Measurements CSV</a>'
        '<a href="coverage.csv">Coverage CSV</a><a href="protocol.json">Protocol JSON</a></nav>',
        '</main><script>'
        'const setting = document.getElementById("setting");'
        'const decoder = document.getElementById("decoder");'
        'const controls = document.getElementById("controls");'
        'const panels = document.querySelectorAll("[data-setting]");'
        'const marks = document.querySelectorAll("[data-method]");'
        'const allCharts = new Map(Array.from(panels, panel => {'
        '  const view = panel.querySelector("[data-chart-view]");'
        '  return [panel, view.cloneNode(true)];'
        '}));'
        'const update = () => {'
        '  const selected = decoder?.value || "";'
        '  panels.forEach(panel => {'
        '    panel.hidden = Boolean(setting && panel.dataset.setting !== setting.value);'
        '    if (panel.hidden) return;'
        '    const view = panel.querySelector("[data-chart-view]");'
        '    if (view.dataset.selection === selected) return;'
        '    const match = Array.from(panel.querySelectorAll("[data-chart-method]"))'
        '      .find(template => template.dataset.chartMethod === selected);'
        '    const source = match ? match.content : allCharts.get(panel);'
        '    view.replaceChildren(...source.cloneNode(true).childNodes);'
        '    view.dataset.selection = selected;'
        '  });'
        '  marks.forEach(mark => {'
        '    const method = mark.dataset.method;'
        '    mark.toggleAttribute("hidden", Boolean(selected && method !== controls.dataset.baseline && method !== selected));'
        '  });'
        '};'
        'if (setting || decoder) controls.hidden = false;'
        'setting?.addEventListener("change", update);'
        'decoder?.addEventListener("change", update);'
        'update();'
        '</script></body></html>',
    ])
    report = run / "index.html"
    report.write_text("\n".join(result), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing metrics.csv, coverage.csv, and protocol.json")
    args = parser.parse_args()
    print(build_report(args.run))


if __name__ == "__main__":
    main()
