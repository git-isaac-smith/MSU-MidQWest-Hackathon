"""Build the self-contained decoder results catalog from archived measurements."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import median
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/figures/Decoder error results.html"
FRONTIER = "experiments/circuit-frontier-01/metrics.csv"
ACCURACY = "experiments/iteration-04/accuracy"
BATH_STRESS = "experiments/bath-background-stress-01"
BATH_RARE = "experiments/bath-rare-confirmation-02"
BATH_CONTROLS = "experiments/bath-clock-controls-02"
BATH_SPECTRUM = "experiments/bath-fft-timing-02"
CATEGORIZATION = "experiments/high-noise-categorization-01"
SCALING = "experiments/qubit-scaling-01"
PROTOTYPES = "new-approach/results"

METHODS = {
    "mwpm": "Ordinary MWPM",
    "ordinary": "Ordinary MWPM",
    "correlated": "Correlated MWPM",
    "correlated_mwpm": "Correlated MWPM",
    "union_find": "Weighted union-find",
    "union_find_handwritten": "Handwritten union-find",
    "fast_hybrid": "Fast hybrid",
    "css_routed": "CSS routed",
    "native": "Standard correlated matching",
    "bath_clock": "Bath-weighted correlated matching",
    "belief": "Belief matching",
    "dispute": "Disagreement search",
    "exact_small": "Exact posterior lookup",
    "frozen": "Distance-based hybrid",
    "clock_corr": "Bath-clock correlated MWPM",
    "clock_ord": "Bath-clock ordinary MWPM",
    "expanded_corr": "Equal-axis correlated MWPM",
    "expanded_ord": "Equal-axis ordinary MWPM",
    "selected_corr": "Bath-selected correlated MWPM",
    "selected_ord": "Bath-selected ordinary MWPM",
    "portable": "Portable posterior lookup",
    "raw_site_parity": "Raw site CNN parity",
    "fft_site_parity": "FFT site CNN parity",
    "raw_site_final": "Raw site CNN + terminal matching",
    "fft_site_final": "FFT site CNN + terminal matching",
    "list_matching": "Matching with alternative corrections",
    "hybrid_raw_learned": "Raw-input logical CNN override",
    "hybrid_learned": "FFT logical CNN override",
    "hybrid_margin_only": "Selective exact posterior · matching margin",
    "hybrid_balanced": "Balanced FFT logical hybrid",
    "hybrid_accuracy": "Selective exact posterior · full shortlist",
    "full_posterior": "Exact posterior on every shot",
    "activity": "Activity routing",
    "gap2": "Logical-gap routing",
    "learned": "Learned routing",
    "learned_rescue": "Learned routing + rescue",
    "cached_teacher": "Cached posterior + selective exact inference",
}


def method(key):
    return METHODS.get(key, key.replace("_", " ").capitalize())


def records(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def document(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pct(probability):
    return f"{float(probability) * 100:g}%"


def case(row):
    return f"d{row['distance']} · {row['basis'].upper()} · p = {pct(row['p'])}"


@dataclass(frozen=True)
class Result:
    condition: str
    method: str
    failures: int | str | None
    shots: int | str | None
    shot_time: float | str | None = None
    rounds: int | str | None = None
    round_time: float | str | None = None
    status: str = ""
    highlight: bool = False

    def render(self):
        if self.shots not in (None, "", 0, "0"):
            failures, shots = int(self.failures), int(self.shots)
            if not 0 <= failures <= shots:
                raise ValueError(f"Invalid counts in {self}")
            rate = f"{100 * failures / shots:.4f}%"
            count = f"{failures:,} / {shots:,}"
        else:
            rate, count = "–", "–"
        per_shot = float(self.shot_time) if self.shot_time not in (None, "") else None
        per_round = self.round_time
        if per_round in (None, "") and per_shot is not None and self.rounds:
            per_round = per_shot / int(self.rounds)
        label = escape(self.method)
        if self.status:
            label += f'<span class="row-note">{escape(self.status)}</span>'
        state = ' class="highlight"' if self.highlight else ""
        timing = "–" if per_shot is None else f"{per_shot:,.3f}"
        normalized = "–" if per_round in (None, "") else f"{float(per_round):,.3f}"
        return (
            f'<tr{state}><td>{escape(self.condition)}</td><th scope="row">{label}</th>'
            f'<td class="rate">{rate}</td><td>{count}</td><td>{timing}</td><td>{normalized}</td></tr>'
        )


@dataclass(frozen=True)
class Study:
    key: str
    title: str
    description: str
    sources: tuple[tuple[str, str], ...]
    rows: list[Result]
    condition_label: str = "Condition"
    expanded: bool = False

    def render(self, output):
        body = "\n".join(row.render() for row in self.rows)
        links = " · ".join(
            f'<a href="{quote(Path(os.path.relpath(ROOT / path, output.parent)).as_posix())}">'
            f'{escape(label)}</a>'
            for label, path in self.sources
        )
        return f"""<section id="{self.key}" class="study">
<details{" open" if self.expanded else ""}>
<summary><h2><span class="study-title">{escape(self.title)}</span>
<span class="row-count">{len(self.rows)} results</span></h2></summary>
<div class="study-content">
<p>{escape(self.description)}</p>
<div class="table-scroll" role="region" aria-label="{escape(self.title)} results" tabindex="0"><table>
<thead><tr><th scope="col">{escape(self.condition_label)}</th><th scope="col">Method</th>
<th scope="col">Logical error</th><th scope="col">Failures / shots</th>
<th scope="col">µs/shot</th><th scope="col">µs/round</th></tr></thead>
<tbody>
{body}
</tbody></table></div>
<p class="sources">{links}</p>
</div></details></section>"""


def circuit_frontier():
    comparisons = {3: "fast_hybrid", 5: "css_routed", 7: "css_routed"}
    primary, archive = [], []
    for row in records(FRONTIER):
        distance = int(row["distance"])
        selected = row["study"] in ("primary", "confirmation") and row["method"] in (
            "mwpm", "correlated", comparisons[distance]
        )
        condition = case(row)
        if not selected:
            variant = row["variant"].replace("_", " ")
            condition += f" · {row['study']} · {variant}"
        result = Result(
            condition, method(row["method"]), row["errors"], row["shots"],
            row["batch_us_shot"], round_time=row["batch_us_round"],
            highlight=selected and row["method"] == comparisons[distance],
        )
        (primary if selected else archive).append(result)
    assert len(primary) == 9
    source = (("Study report", "experiments/circuit-frontier-01/results.md"), ("Metrics CSV", FRONTIER))
    return (
        Study("reference", "Circuit-level reference",
              "Held-out rotated Z memory with simple_uniform p = 0.5% and rounds equal to distance. "
              "Each group uses the same shots. Fast hybrid is the primary d3 candidate and CSS routing is the d5 and d7 candidate.",
              source, primary, expanded=True),
        Study("frontier", "Circuit-level frontier sweep",
              "Remaining rows from the circuit-level frontier study, including exploratory methods on held-out cases, "
              "the ordinary physical-noise sweep, and final-round measurement, measurement-burst, and spatially correlated stress variants.",
              source, archive),
    )


def bath_stress():
    path = f"{BATH_STRESS}/result.json"
    timing_path = f"{BATH_STRESS}/timing/result.json"
    points = document(path)["points"]
    timings = {}
    sources = [("Study report", f"{BATH_STRESS}/README.md"), ("Counts and seeds", path)]
    if (ROOT / timing_path).is_file():
        timings = {(r["p_check"], r["arm"]): r for r in document(timing_path)["points"]}
        if len(timings) != len(points):
            raise ValueError("Bath timing archive must cover all stress points")
        sources.append(("Decode timing and verification", timing_path))
    rows = []
    for point in points:
        arm = point["arm"]
        timing = timings[(point["p_check"], arm)] if timings else {}
        if timing and any(timing[k] != point[k] for k in ("shots", "logical_failures")):
            raise ValueError(f"Bath timing predictions differ at {point['p_check']}, {arm}")
        rows.append(Result(
            pct(point["p_check"]),
            "Frame-calibrated clock" if arm.startswith("Clock") else "Fitted balanced check",
            point["logical_failures"], point["shots"], timing.get("batch_us_shot"),
            round_time=timing.get("batch_us_round"),
        ))
    assert len(rows) == 10
    description = (
        "Distance-7 Z memory, seven rounds, five fixed bath layouts. Each round adds the listed error probability "
        "independently on the X and Z axes at every data site. Reset and readout stay at 0.1%. Both arms use erasure flags and have "
        "their own simulated physical channels."
    )
    if timings:
        description += " Timing covers warmed decoding of the archived records and includes flag-conditioned work."
    return Study("bath-stress", "Bath background-check stress", description, tuple(sources), rows,
                 condition_label="Added error per axis", expanded=True)


def bath_timing_candidates():
    rare_path = f"{BATH_RARE}/result.json"
    rare = document(rare_path)
    condition = f"d{rare['distance']} · {rare['basis'].upper()} · P_CHECK = {pct(rare['p_check'])}"
    lower, upper = rare["wilson_95"]
    yield Study(
        "bath-rare", "Fitted balanced-check confirmation",
        "Fresh distance-7 Z memory shots at nominal clock period 6. The interval describes Monte Carlo precision "
        "for the five fixed layouts. The independent serial timing is archived with the original 20,000-shot bath stress row.",
        (("Confirmation report", f"{BATH_RARE}/README.md"), ("Counts, intervals, and seeds", rare_path),
         ("Original serial timing", f"{BATH_STRESS}/timing/README.md")),
        [Result(condition, "Individual fitted balanced check", rare["logical_failures"], rare["shots"],
                status=f"Fit-only calibration · 95% Wilson interval {lower:.5%}–{upper:.5%}", highlight=True)],
        expanded=True,
    )

    clock_path = f"{BATH_CONTROLS}/result.json"
    fft_path = f"{BATH_CONTROLS}/fft_result.json"
    timing_path = f"{BATH_CONTROLS}/timing.json"
    controls = {
        "fixed_6": ("Fixed clock at 6", "Author’s true-parameter frame and prior"),
        "global": ("Global clock, independent calibration", "One time from 400 training bath fits"),
        "global_installed_calibration": ("Global clock, installed calibration", "One time from fitted models of 245 installed sites"),
        "shuffled": ("Shuffled fitted check times", "Individual times permuted within each layout"),
        "individual": ("Individual fitted balanced check", "Fit-only calibration for times, frames, and priors"),
    }
    spectrum = {
        "fft_timing": ("Full-spectrum FFT check timing", "FFT times with local fitted frames and priors"),
        "dominant_tone_timing": ("Dominant-tone check timing", "Single-tone times with local fitted frames and priors"),
        "fft_only": ("End-to-end full-spectrum FFT", "FFT times, frames, and priors from simulated calibration records"),
    }
    timings = {}
    if (ROOT / timing_path).is_file():
        timing_archive = document(timing_path)
        timed = timing_archive["arms"]
        timings = {arm["arm"]: arm for arm in timed}
        if len(timed) != len(timings) or timings.keys() != (controls.keys() | spectrum.keys()):
            raise ValueError(f"Unexpected arm coverage in {timing_path}")
        prefix_shots = timing_archive["prefix_shots_per_layout"]
        timed_calls = timing_archive["timed_calls_per_layout"]
    studies = (
        ("bath-clock-controls", "Bath clock schedule controls", clock_path, controls,
         "Each arm uses 20,000 separately sampled shots across the same five layouts. The global, shuffled, and individual "
         "schedules compute local frames and priors from bath fits at the assigned times.",
         (("Schedule controls report", f"{BATH_CONTROLS}/README.md"), ("Counts and intervals", clock_path),
          ("Frozen schedule calibration", f"{BATH_CONTROLS}/calibration.json"))),
        ("bath-fft-controls", "Bath FFT schedule and calibration", fft_path, spectrum,
         "Each arm uses 20,000 separately sampled shots across the same five layouts and the same herald-conditioned "
         "matching algorithm. The rail calibration traces are noiseless simulations.",
         (("Spectrum reconstruction report", f"{BATH_SPECTRUM}/README.md"),
          ("Logical replay methods", f"{BATH_CONTROLS}/README.md"), ("Counts and intervals", fft_path),
          ("Frozen FFT calibration", f"{BATH_CONTROLS}/fft_calibration.json"))),
    )
    for key, title, path, candidates, description, sources in studies:
        measured = document(path)["arms"]
        if len(measured) != len(candidates) or {arm["arm"] for arm in measured} != candidates.keys():
            raise ValueError(f"Unexpected arm coverage in {path}")
        if any(arm["shots"] != 20_000 for arm in measured):
            raise ValueError(f"Expected 20,000 shots per arm in {path}")
        rows = []
        for arm in measured:
            label, status = candidates[arm["arm"]]
            timing = timings.get(arm["arm"], {})
            rows.append(Result(condition, label, arm["logical_errors"], arm["shots"], timing.get("batch_us_shot"),
                               round_time=timing.get("batch_us_round"), status=status))
        if timings:
            description += (f" Serial decode timing uses the first {prefix_shots:,} shots per layout, for "
                            f"{5 * prefix_shots:,} shots per arm. It averages the layout medians from "
                            f"{timed_calls} calls per batch after warmup.")
            sources += (("Serial decode timing and replay protocol", timing_path),)
        yield Study(key, title, description, sources, rows, expanded=True)


def fixed_stim(density=False):
    directory = f"{CATEGORIZATION}/fixed-stim"
    path = f"{directory}/{'density' if density else 'scenarios'}.csv"
    rows = [
        Result(case(row) + (f" · {row['density_bin']}" if density else ""), method(key),
               row[f"{key}_errors"], row["shots"], row.get(f"{key}_batch_us_per_shot"), row["rounds"])
        for row in records(path) for key in ("ordinary", "correlated")
    ]
    if density:
        return Study("density", "Fixed Stim results by detector density",
                     "The fixed-Stim shots grouped by the fraction of the circuit’s 336 detectors that fired. "
                     "Only nonempty bins appear. The archive times complete runs.", (("Density CSV", path),), rows)
    assert len(rows) == 20
    return Study("fixed-stim", "Fixed Stim high-noise sweep",
                 "Distance-7 X and Z memory, seven rounds. The same p sets gate depolarization, round-start data "
                 "depolarization, reset flips, and readout flips. Each method decodes the same records within a setting.",
                 (("Study report", f"{directory}/README.md"), ("Scenario CSV", path)), rows, expanded=True)


def source_removal():
    directory = f"{CATEGORIZATION}/fault-sources"
    path = f"{directory}/removals.csv"
    summary = f"{directory}/summary.json"
    labels = {"gate": "After-Clifford gate faults", "round_start_data": "Round-start data faults",
              "readout": "Before-measurement flips", "reset": "After-reset flips"}
    rows, checks = [], []
    removals = defaultdict(list)
    for row in records(path):
        removals[float(row["p"])].append(row)
    for setting in document(summary)["cases"]:
        condition = f"d7 · Z · p = {pct(setting['p'])}"
        rows.append(Result(condition, "Correlated matching · all faults", setting["full_failures"], setting["shots"]))
        checks.append(Result(condition, "Correlated matching · native Stim sample",
                             setting["direct_check_failures"], setting["shots"]))
        for row in removals[setting["p"]]:
            rows.append(Result(condition, f"Remove {labels[row['removed']].lower()}",
                               row["remaining_failures"], row["shots"]))
    assert len(rows) == 15
    sources = (("Paired removal CSV", path), ("Model verification", summary))
    return (
        Study("fault-removal", "Stim fault-family removal",
              "Correlated matching uses the complete circuit’s graph for every row. Each removal keeps the other "
              "fault samples fixed. The simulator can remove overlapping contributors to the same logical failure.",
              sources, rows, expanded=True),
        Study("fault-control", "Native Stim check for fault composition",
              "Fresh native Stim samples verify the separate fault-family simulation at the same three noise settings.",
              (("Model verification", summary),), checks),
    )


def circuit_rows(path, names=None, use_prior=False):
    rows = []
    for row in records(path):
        key = row["decoder"]
        label = (names or {}).get(key, method(key))
        if use_prior:
            label += " · " + row["prior"].replace("_", " ").capitalize()
        rows.append(Result(case(row), label, row["failures"], row["shots"],
                           row["batch_us_per_shot"], row["rounds"]))
    return rows


def hybrid():
    path = f"{ACCURACY}/summary.json"
    test = f"{ACCURACY}/test.csv"
    archive = circuit_rows(test, {"exact_small": "Distance-based hybrid · exact posterior",
                                  "dispute": "Distance-based hybrid · disagreement search"})
    pooled = [
        Result("24 scenarios", method(key), row["failures"], row["shots"], row["batch_us_per_shot"],
               highlight=key == "frozen")
        for key, row in document(path).items()
    ]
    assert len(archive) == 96 and len(pooled) == 4
    return (
        Study("hybrid", "Distance-based hybrid · 24-scenario test",
              "Pooled holdout on 480,000 detector-only shots. The equally weighted grid spans d3, d5, and d7 in X "
              "and Z memory at p = 0.1%, 0.3%, 0.5%, and 1%. The hybrid uses an exact lookup at d3 and disagreement "
              "search at d5 and d7.",
              (("Held-out study", "Iteration 04.md"), ("Pooled counts and paired comparisons", path)),
              pooled, expanded=True),
        Study("hybrid-grid", "Distance-based hybrid by circuit setting",
              "All 24 scenarios from the 480,000-shot holdout. Each circuit contributes 20,000 shots.",
              (("Held-out circuit CSV", test),), archive),
    )


def additional_circuit_studies():
    setups = (
        ("earliest", "Initial matching benchmark", "The original ordinary-matching starter study.",
         "results/iteration-01/benchmark.csv", False),
        ("adaptive", "Adaptive matching and early circuit sweep",
         "Iteration 02 paired ordinary and adaptive or bath-initialized matching runs.",
         "experiments/iteration-02/sweep/benchmark.csv", False),
        ("transfer", "Initial bath-prior transfer",
         "Fixed Stim records decoded with native, expanded equal-axis, and bath-derived matching weights.",
         "experiments/iteration-03/bath-transfer/benchmark.csv", True),
        ("transfer-replication", "Bath-prior transfer replication",
         "A fresh fixed-circuit grid repeating the bath-prior transfer comparison.",
         "experiments/iteration-03/bath-transfer-replication/benchmark.csv", True),
        ("bath-confirmation", "Bath-prior confirmation",
         "The 3,000,000-shot fixed-circuit study of ordinary and correlated graph priors.",
         "experiments/bath-confirmation-01/run/benchmark.csv", True),
        ("stim-archive", "Stim shot archive",
         "A separate saved detector-shot archive, with all 24 circuits and three decoders.",
         "experiments/iteration-04/stim-shot-archive/stats.csv", False),
        ("screen", "Verified 240,000-shot decoder replay",
         "A separate saved detector-shot run with 10,000 shots per circuit. The pooled hybrid count is 4,660.",
         "experiments/iteration-04/screen-run-954131/stats.csv", False),
    )
    for key, title, description, path, use_prior in setups:
        yield Study(key, title, description, (("Archived CSV", path),),
                    circuit_rows(path, use_prior=use_prior))


def bath_archive():
    replay = "experiments/sakidja-table5-replay-01/result.json"
    archive = document(replay)
    rows = [
        Result(f"{label} · d7 · Z · 0.1% per axis", arm["arm"], arm[key], archive["shots_per_arm"])
        for key, label in (("published_logical_errors", "Published"), ("replay_logical_errors", "Fresh replay"))
        for arm in archive["arms"]
    ]
    yield Study("table-five", "Sakidja Table 5 replay",
                "Published and fresh counts for the native bath setting. The two timing arms have separate simulator "
                "shots and receive erasure locations.",
                (("Replay report", "experiments/sakidja-table5-replay-01/README.md"), ("Replay JSON", replay)), rows)
    directory = "experiments/bath-basis-policy-01/confirmation"
    path = f"{directory}/scenarios.csv"
    summary = f"{directory}/summary.json"
    labels = {"policy": "Bath in X, equal-axis in Z", "equal": "Always equal-axis", "bath": "Always bath"}
    pooled = document(summary)["pooled"]
    rows = [
        Result("24 scenarios", labels[key], row["failures"], pooled["shots"], row["batch_us_per_shot"])
        for key, row in pooled["methods"].items()
    ]
    rows.extend(
        Result(case(row), label, row[f"{key}_failures"], row["shots"],
               float(row[f"{key}_batch_s"]) * 1e6 / int(row["shots"]), row["rounds"])
        for row in records(path) for key, label in labels.items()
    )
    yield Study("bath-policy", "Basis-specific bath prior",
                "Fresh fixed Stim circuits with the decoder prior frozen before sampling. Policy timing uses the "
                "selected prior’s measured batch time, since the circuit basis is known in advance.",
                (("Study report", "experiments/bath-basis-policy-01/README.md"), ("Scenario CSV", path)), rows)


def scaling():
    path = f"{SCALING}/metrics.csv"
    rows = []
    for row in records(path):
        note = ""
        if row["status"] != "ok":
            note = row["status"].replace("_", " ").capitalize()
            if row["reason"]:
                note += f" · {row['reason']}"
        rows.append(Result(case(row) + f" · {int(row['data_qubits']):,} data qubits", method(row["method"]),
                           row["failures"], row["shots"], row["batch_us_per_shot"], row["rounds"], status=note))
    yield Study("scaling", "Circuit-level scaling through d31",
                "35 decoder configurations, d3/5/7/15/31 in X and Z, up to 961 data qubits and 1,921 active physical "
                "qubits. Each case sampled 10,000 shots. Predeclared shorter prefixes and resource limits are marked.",
                (("Scaling report", f"{SCALING}/report.md"), ("Scaling CSV", path)), rows)
    path = f"{SCALING}/high-noise/measured/metrics.csv"
    rows = [
        Result(f"d{row['distance']} · Z · p = 1% · {row['data_qubits']} data qubits", method(row["method"]),
               row["errors"], row["shots"], row["batch_us_per_shot"], row["distance"])
        for row in records(path)
    ]
    yield Study("scaling-stress", "Higher-noise scaling through d31",
                "Independent 2,048-shot Stim records per case, with rounds equal to distance.",
                (("Higher-noise report", f"{SCALING}/high-noise/README.md"), ("Metrics CSV", path)), rows)
    path = f"{SCALING}/heralds/measured/accuracy-timing.csv"
    rows = [
        Result(f"d{r['distance']} · Z · {r['clock']} at t = {float(r['time']):g}",
               f"{key.capitalize()} matching", r[f"{key}_failures"], r["shots"], r[f"{key}_us_per_shot"], 1)
        for r in records(path) for key in ("blind", "heralded")
    ]
    yield Study("heralds", "Heralded bath control through d31",
                "One effective bath channel and one round of ideal checks per shot. The heralded matcher sees site "
                "flags and rebuilds the graph during inference. The blind matcher sees the check parities.",
                (("Heralded study report", f"{SCALING}/heralds/README.md"), ("Accuracy and timing CSV", path)), rows)


def spectral():
    directory = f"{SCALING}/fft/measured"
    accuracy = f"{directory}/accuracy.csv"
    timing = f"{directory}/timing.csv"
    times = {(r["distance"], r["method"]): r["median_batch_us_per_shot"] for r in records(timing)}
    rows = [
        Result(f"d{r['distance']} · Z · {r['data_qubits']} data qubits", method(r["method"]),
               r["failures"], r["shots"], times[(r["distance"], r["method"])], 5)
        for r in records(accuracy)
    ]
    yield Study("fft-scaling", "Five-round FFT and posterior scaling through d33",
                "Controlled Z memory. Each of five rounds applies independent 0.8% data X faults and 4% check "
                "readout flips. Final data readout is ideal. Timing uses warmed 64-record batches.",
                (("FFT scaling report", f"{SCALING}/fft/README.md"), ("Accuracy CSV", accuracy),
                 ("Timing CSV", timing)), rows)
    for key, title in (
        ("hybrid-01", "Balanced FFT hybrid"),
        ("full-map-01", "Compiled full posterior"),
        ("distance7-01", "Distance-7 selective posterior"),
        ("list-matching-01", "Distance-7 matching shortlist"),
    ):
        directory = f"{PROTOTYPES}/{key}"
        path = f"{directory}/results.csv"
        raw = f"{directory}/results.json"
        stage_times = document(raw)["timings"] if key in ("distance7-01", "list-matching-01") else []
        mode_labels = {
            "distance7-01": {"aggressive": "Selective posterior · every proposed override",
                            "selected": "Selected policy · plain MWPM"},
            "list-matching-01": {"aggressive": "Matching shortlist · every proposed override",
                                "selected": "Matching shortlist · selected cutoff"},
        }
        rows = []
        for row in records(path):
            if row["seed"] or (row.get("scope") and row["scope"] != "pooled"):
                continue
            throughput = row.get("shots_per_second")
            shots = row.get("total") or row["shots"]
            time = 1e6 / float(throughput) if throughput else None
            if stage_times:
                routed = row["mode"] != "mwpm" and float(row["query_percent"]) > 0
                stages = ["mwpm_seconds"] + (["margin_seconds", "posterior_seconds"] if routed else [])
                time = sum(rep[stage] for rep in stage_times for stage in stages) * 1e6 / int(shots)
            rows.append(Result("d5 · Z" if row.get("scope") else "d7 · Z",
                               row.get("label") or mode_labels.get(key, {}).get(row["mode"], method(row["mode"])),
                               row["errors"], shots, time, 5))
        description = "Separate holdout under the controlled five-round model with 0.8% data X faults and 4% check readout flips."
        if stage_times:
            description += " Timing averages the inference stages used by each method over the holdout."
        yield Study(key, title, description, (("Pooled and seed results", path), ("Run and timing details", raw)), rows)
    path = f"{PROTOTYPES}/cached-teacher-01/results.json"
    cached = document(path)
    rows = [
        Result("d5 · Z", method(key), row["errors"], cached["accuracy"]["shots"],
               cached["timing"]["median_seconds"][key] * 1e6 / cached["timing"]["shots_per_pass"], 5)
        for key, row in cached["accuracy"]["error_rates"].items()
    ]
    yield Study("cached-teacher", "Cached posterior and selective exact inference",
                "Separate five-round controlled holdout. Timing is the median of nine 49,152-shot passes.",
                (("Study report", f"{PROTOTYPES}/cached-teacher-01/README.md"), ("Accuracy and timing JSON", path)), rows)
    path = f"{PROTOTYPES}/screen-run-2026112102/stim-comparison/error-percentages.csv"
    rows = [Result("d5 · Z", row["method"], row["errors"], row["total"])
            for row in records(path) if row["metric"] == "logical_shot"]
    yield Study("fft-site", "Early FFT site decoder",
                "Complete-shot logical outcomes on the controlled circuit used to compare FFT filtering with matching.",
                (("Logical and site error CSV", path),), rows)


def bursts_and_compute():
    path = "experiments/spatial-burst-01/results.csv"
    rows = [
        Result(f"d{r['distance']} · Z · density {r['density']} · ξ = {r['xi']} · {r['exposure']}",
               "Ordinary MWPM", r["errors"], r["shots"], r["batch_us_shot"], r["rounds"])
        for r in records(path)
    ]
    yield Study("bursts", "Spatially correlated bursts",
                "Independent circuit-level background p = 0.5% with extra spatial data faults. The exposure "
                "and ξ fields identify distinct burst regimes.",
                (("Burst study report", "experiments/spatial-burst-01/report.md"), ("Results CSV", path)), rows)
    directory = "experiments/compute-frontier-02"
    accuracy, timing = f"{directory}/accuracy.csv", f"{directory}/timing.csv"
    times = defaultdict(list)
    for row in records(timing):
        if row["measurement"] == "batch":
            times[row["distance"], row["method"]].append(float(row["microseconds_per_shot"]))
    rows = []
    for row in records(accuracy):
        samples = times[row["distance"], row["method"]]
        rows.append(Result(f"d{row['distance']} · Z · p = 0.5%", method(row["method"]),
                           row["errors"], row["shots"], median(samples) if samples else None, row["distance"]))
    yield Study("compute", "Circuit-level compute study",
                "Fixed circuit-level noise, 300,000 shots per distance, and median warm serial batch timing.",
                (("Compute study report", f"{directory}/summary.md"), ("Accuracy CSV", accuracy), ("Timing CSV", timing)), rows)
    for directory, title in (
        ("specialist-budget-01", "Specialist iteration budget"),
        ("specialist-budget-accuracy-01", "Specialist budget accuracy follow-up"),
    ):
        report = f"experiments/{directory}/results.md"
        path = f"experiments/{directory}/confirmation.csv"
        rows = [
            Result("d7 · Z · p = 0.5%", method(r["method"]), r["failures"], r["shots"], r["median_us_per_shot"], 7)
            for r in records(path)
        ]
        yield Study(directory, title, "Fresh paired confirmation on the fixed circuit and warmed serial timing.",
                    (("Study report", report), ("Confirmation CSV", path)), rows)


def catalog():
    reference, frontier = circuit_frontier()
    fault_removal, fault_control = source_removal()
    pooled_hybrid, hybrid_grid = hybrid()
    featured = [*bath_timing_candidates(), reference, bath_stress(), fixed_stim(), fault_removal, pooled_hybrid]
    archive = [
        hybrid_grid, *scaling(), *spectral(), *bursts_and_compute(), frontier, fixed_stim(density=True),
        fault_control, *bath_archive(), *additional_circuit_studies(),
    ]
    return featured, archive


STYLE = """
:root {
  color-scheme: light;
  --ink: #203449;
  --muted: #516578;
  --rule: #d9e4e9;
  --teal: #00756e;
  --teal-dark: #065750;
  --tint: #e8f6f3;
  --light: #f6f9fb;
  font: 15px/1.55 "Segoe UI", system-ui, sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: white; }
a { color: var(--teal-dark); text-underline-offset: 3px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 3px solid var(--teal); outline-offset: 3px; }
h1, h2, p { margin: 0; }
header { border-top: 6px solid var(--teal); border-bottom: 1px solid var(--rule); }
.frame { width: min(1080px, calc(100% - 56px)); margin: auto; }
.header-content { padding: 37px 0 30px; }
.headline { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px 26px; }
.report-links { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
h1 { color: #183048; font-size: clamp(2.12rem, 4.8vw, 3.65rem); font-weight: 720; line-height: 1.08; letter-spacing: -.045em; }
.intro { max-width: 75ch; margin-top: 16px; color: var(--muted); }
.download { border: 1px solid #aacdc9; border-radius: 5px; color: var(--teal-dark); padding: 7px 12px; font-weight: 650; }
main { padding: 24px 0 45px; }
nav { display: flex; flex-wrap: wrap; gap: 2px 20px; padding-bottom: 13px; border-bottom: 1px solid var(--rule); }
nav a { font-size: .89rem; font-weight: 650; }
.controls { padding: 15px 0 17px; display: flex; align-items: end; gap: 12px; flex-wrap: wrap; }
.search { display: grid; gap: 4px; flex: 1 1 300px; max-width: 550px; }
.search label { font-size: .89rem; font-weight: 650; }
input { width: 100%; font: inherit; padding: 7px 10px; border: 1px solid #9fb1bf; border-radius: 4px; color: var(--ink); }
.search-status { min-height: 26px; font-size: .86rem; color: var(--muted); }
.study { margin-bottom: 13px; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
summary { cursor: pointer; list-style: none; display: flex; align-items: center; gap: 10px; padding: 12px 16px; background: var(--light); }
summary::-webkit-details-marker { display: none; }
summary::before { content: ""; flex: 0 0 8px; width: 8px; height: 8px; border-right: 2px solid var(--teal); border-bottom: 2px solid var(--teal); transform: rotate(-45deg); margin-right: 4px; }
details[open] > summary::before { transform: rotate(45deg) translateY(-2px); }
details[open] > summary { border-bottom: 1px solid var(--rule); }
summary:hover { background: #edf6f6; }
summary:focus-visible { outline-offset: -3px; }
summary h2 { flex: 1; display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 1.15rem; font-weight: 680; line-height: 1.3; }
.row-count { color: var(--muted); font-size: .76rem; white-space: nowrap; font-weight: 400; font-variant-numeric: tabular-nums; }
.study-content > p { padding: 11px 16px; color: var(--muted); font-size: .88rem; }
.study-content > .sources { padding: 9px 16px; border-top: 1px solid var(--rule); }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .89rem; }
thead { background: #eaf0f4; color: #2b4459; }
thead th { font-weight: 680; letter-spacing: .012em; }
th, td { padding: 7px 11px; vertical-align: top; text-align: left; }
tr > :first-child { padding-left: 16px; }
tr > :last-child { padding-right: 16px; }
tr > :nth-child(n+3) { text-align: right; white-space: nowrap; }
tbody tr { border-top: 1px solid #e6edf1; }
tbody tr:nth-child(even) { background: #f9fbfc; }
tbody th { font-weight: 600; }
tbody td:first-child { color: var(--muted); }
tbody .rate { color: var(--teal-dark); font-weight: 690; }
tbody tr.highlight { background: var(--tint); }
.row-note { display: block; color: var(--muted); font-size: .78rem; font-weight: 400; }
.archive-title { font-size: 1.5rem; margin: 27px 0 10px; letter-spacing: -.02em; }
footer { padding: 16px 0 24px; border-top: 1px solid var(--rule); }
footer p { font-size: .85rem; color: var(--muted); }
[hidden] { display: none !important; }
@media (max-width: 620px) {
  .frame { width: calc(100% - 28px); }
  .header-content { padding: 22px 0 20px; }
  main { padding-top: 16px; }
  th, td { padding: 7px 8px; }
  tr > :first-child { padding-left: 10px; }
  tr > :last-child { padding-right: 10px; }
}
@media print {
  header { border-top-width: 2px; }
  .controls, nav, .report-links { display: none; }
  .frame { width: 100%; }
  .table-scroll { overflow: visible; }
  tr { break-inside: avoid; }
}
"""

SEARCH = r"""
const input = document.querySelector("#filter");
const status = document.querySelector("#search-status");
const archive = document.querySelector("#archive");
const studies = Array.from(document.querySelectorAll(".study"), section => ({
  section,
  details: section.querySelector("details"),
  header: section.querySelector(".study-title").textContent,
  description: section.querySelector(".study-content > p").textContent,
  count: section.querySelector(".row-count"),
  rows: Array.from(section.querySelectorAll("tbody tr")),
}));
const initialOpen = new Map();
let wasFiltering = false;
input.addEventListener("input", () => {
  const terms = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const filtering = terms.length > 0;
  if (filtering && !wasFiltering) {
    studies.forEach(study => initialOpen.set(study.section.id, study.details.open));
  }
  let hits = 0;
  studies.forEach(study => {
    const context = (study.header + " " + study.description).toLowerCase();
    const matches = study.rows.filter(row => {
      const text = context + " " + row.textContent.toLowerCase();
      const condition = row.cells[0].textContent.toLowerCase();
      row.hidden = !terms.every(term => /^d\d+$/.test(term)
        ? condition.split(/\b/).includes(term)
        : text.includes(term));
      return !row.hidden;
    }).length;
    study.count.textContent = matches + (matches === 1 ? " result" : " results");
    study.section.hidden = matches === 0;
    if (filtering) {
      study.details.open = matches > 0;
    } else if (wasFiltering) {
      study.details.open = initialOpen.get(study.section.id);
    }
    hits += matches;
  });
  archive.hidden = !archive.querySelector(".study:not([hidden])");
  status.textContent = filtering ? hits + (hits === 1 ? " matching row" : " matching rows") : "";
  wasFiltering = filtering;
});
document.querySelectorAll("nav a").forEach(link => link.addEventListener("click", () => {
  if (input.value) {
    input.value = "";
    input.dispatchEvent(new Event("input"));
  }
  const details = document.querySelector(link.getAttribute("href") + " details");
  if (details) details.open = true;
}));
"""


def render(output):
    featured, archive = catalog()
    links = (
        ("bath-rare", "Fitted confirmation"), ("bath-clock-controls", "Clock controls"), ("bath-fft-controls", "FFT calibration"),
        ("reference", "Circuit reference"), ("bath-stress", "Bath stress"), ("fixed-stim", "Fixed Stim"),
        ("fault-removal", "Fault removal"), ("hybrid", "Hybrid test"), ("archive", "Study archive"),
    )
    navigation = "\n".join(f'<a href="#{key}">{label}</a>' for key, label in links)
    featured_tables = "\n".join(study.render(output) for study in featured)
    archive_tables = "\n".join(study.render(output) for study in archive)
    pdf_dir = Path(os.path.relpath(ROOT / "output/pdf", output.parent))
    bath_pdf = quote((pdf_dir / "Decoder error rates - bath timing.pdf").as_posix())
    original_pdf = quote((pdf_dir / "Decoder error rates.pdf").as_posix())
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decoder error results | MidQWest</title>
<style>{STYLE}</style>
</head>
<body>
<header>
  <div class="frame header-content">
    <div class="headline"><h1>Decoder error results</h1>
      <div class="report-links"><a class="download" href="{bath_pdf}">Bath timing report PDF</a>
      <a href="{original_pdf}">Original results PDF</a></div>
    </div>
    <p class="intro">Logical failures count wrong predictions per complete memory shot. Compare methods within a
    shared circuit and shot set. µs/round is decoder time divided by the known round count and describes buffered
    throughput. A dash marks an unavailable value.</p>
  </div>
</header>
<main class="frame">
<nav aria-label="Jump to results">
{navigation}
</nav>
<div class="controls">
  <div class="search">
    <label for="filter">Search all results</label>
    <input id="filter" type="search" placeholder="Try d31 projected or heralded" autocomplete="off">
  </div>
  <div id="search-status" class="search-status" role="status"></div>
</div>
{featured_tables}
<section id="archive" aria-labelledby="archive-title">
<h2 id="archive-title" class="archive-title">Study archive</h2>
{archive_tables}
</section>
</main>
<footer class="frame">
<p>Timing protocols vary by study. Zero observed failures are sample results. Source tables provide shot counts,
uncertainty intervals where recorded, and timing details.</p>
</footer>
<script>{SEARCH}</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return sum(len(study.rows) for study in featured + archive), len(featured) + len(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output.resolve()
    rows, studies = render(output)
    print(f"Wrote {rows:,} results across {studies} studies to {output}")


if __name__ == "__main__":
    main()
