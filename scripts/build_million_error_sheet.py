"""Render the million-shot study PDF and exact-count CSV from published results and durable checkpoints."""
import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FIXED = EXP / "million-fixed-03"
BATH = EXP / "million-bath-03"
CONTROLS = EXP / "million-controls-03"
TARGET = 1_000_000
PDF = ROOT / "output/pdf/Decoder error rates - one million shots.pdf"
CSV = FIXED / "combined-results.csv"
BATH_PDF = ROOT / "output/pdf/Decoder error rates - one million bath shots.pdf"
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("million_fixed_plan", FIXED / "run.py")
fixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixed)

NAMES = {
    "mwpm": "Ordinary MWPM", "correlated": "Correlated MWPM", "css_routed": "Selective correlated",
    "fast_hybrid": "Posterior lookup", "expanded_corr": "Equal-axis correlated", "clock_corr": "Clock-prior correlated",
    "basis_select": "Basis-selected prior", "union_find": "Union-find", "native": "Correlated MWPM", "bath_clock": "Clock-prior correlated",
    "belief": "Belief decoder", "hybrid": "Distance-based hybrid", "equal": "Equal-axis correlated",
    "full": "All fault families", "gate": "Remove gate noise", "round_start_data": "Remove round-start data noise",
    "readout": "Remove readout flips", "reset": "Remove reset flips",
    "fixed_6": "Fixed clock", "individual": "Per-qubit fitted time", "global": "One training-bath time",
    "global_installed_calibration": "One installed-bath time", "shuffled": "Shuffled per-qubit times",
    "fft_timing": "FFT time, fitted frame", "dominant_tone_timing": "Single-tone time, fitted frame",
    "fft_only": "FFT time, frame and weights",
}
POLICIES = ["fixed_6", "global", "global_installed_calibration", "shuffled", "individual",
            "fft_timing", "dominant_tone_timing", "fft_only"]
BG = (.001, .005, .01, .02, .04)
FAULT = ("full", "gate", "round_start_data", "readout", "reset")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def interval(w, n, exact):
    if n == 0:
        return None, None
    if exact:
        return (float(beta.ppf(.025, w, n-w+1)) if w else 0.,
                float(beta.ppf(.975, w+1, n-w)) if w < n else 1.)
    z = 1.959963984540054
    p = w / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z/den*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0., center-half) if w else 0., min(1., center+half) if w < n else 1.


def row(group, d, basis, p, method, target, shots, wrong, source, *, ns=None, rounds_sum=None,
        archived=None, warmed=None, exact=False, saved_ci=None, recovered=None, introduced=None):
    if not 0 <= wrong <= shots <= target:
        raise ValueError(f"Invalid totals for {group}/{method}: {wrong}, {shots}, {target}")
    lo, hi = saved_ci if saved_ci is not None else interval(wrong, shots, exact)
    us_shot = warmed["us_per_shot"] if warmed else ns / 1000 / shots if ns and shots else None
    us_round = warmed["us_per_round"] if warmed else ns / 1000 / rounds_sum if ns and rounds_sum else None
    return dict(group=group, distance=d, basis=basis, probability=p, method=method, method_label=NAMES[method],
                target_shots=target, shots=shots, wrong=wrong if shots else None,
                ler=wrong/shots if shots else None, lower_95=lo, upper_95=hi,
                interval="Clopper-Pearson" if exact else ("pooled Wilson approximation" if group == "pooled_total" else "Wilson"), decode_ns=ns if shots else None,
                us_per_shot=us_shot, us_per_round=us_round,
                timing_source=("accelerated 512-shot warmed replay under shared load" if group == "controls"
                               else "campaign inference-only batch wall clock"),
                archived_us_per_shot=archived["batch_us_shot"] if archived else None,
                archived_us_per_round=archived["batch_us_round"] if archived else None,
                archived_timing_source=("experiments/bath-background-stress-01/timing/result.json" if group == "bath"
                                        else "experiments/bath-clock-controls-02/timing.json" if group == "controls" else ""),
                recovered=recovered, introduced=introduced, source=source)


def fixed_rows():
    result = []
    for group in ("grid", "pooled", "fault", "large"):
        for job in fixed.plan(group):
            _, d, b, p, m, target = job
            names = [m]
            if group == "grid":
                names = ["mwpm", "correlated"]
                if b == "z" and p == .005:
                    names += ["fast_hybrid" if d == 3 else "css_routed"]
            elif group == "fault":
                names = FAULT
            elif group == "pooled":
                names = ("native", "bath_clock", "belief", "hybrid", "equal")
            path = fixed.file_for(job)
            chunks = []
            if path.exists():
                state = read(path)
                if tuple(state["job"]) != job:
                    raise ValueError(f"Job plan has changed: {path}")
                chunks = state["chunks"]
            n = sum(c["shots"] for c in chunks)
            for name in names:
                metrics = [c["methods"][name] for c in chunks]
                sums = {k: sum(v[k] for v in metrics) for k in ("ns", "wrong", "recovered", "introduced")}
                paired = group != "large" and bool(n)
                result.append(row(group, d, b.upper(), p, name, target, n, sums["wrong"],
                                  path.relative_to(ROOT).as_posix(), ns=sums["ns"], rounds_sum=n*d,
                                  recovered=sums["recovered"] if paired else None,
                                  introduced=sums["introduced"] if paired else None))
    for name in ("native", "bath_clock", "belief", "hybrid", "equal", "basis_select"):
        records = [r for r in result if r["group"] == "pooled" and
                   r["method"] == (("bath_clock" if r["basis"] == "X" else "equal")
                                    if name == "basis_select" else name)]
        def total(key):
            return sum(r[key] or 0 for r in records)
        result.append(row("pooled_total", "", "X/Z", None, name, total("target_shots"), total("shots"), total("wrong"),
                          "experiments/million-fixed-03/checkpoints/pooled", ns=total("decode_ns"),
                          rounds_sum=sum(r["distance"]*r["shots"] for r in records),
                          recovered=total("recovered") if total("shots") else None,
                          introduced=total("introduced") if total("shots") else None))
    return result


def bath_rows():
    rows = []
    t = read(EXP / "bath-background-stress-01/timing/result.json")["points"]
    timing = {("fixed_6" if x["arm"].startswith("Clock") else "individual", x["p_check"]): x for x in t}
    path = BATH / "result.json"
    published = {(r["arm_id"], r["p_check"]): r for r in read(path)["points"]} if path.exists() else {}
    native_parts = {}
    for p in BG:
        for method in ("fixed_6", "individual"):
            if (method, p) in published:
                r = published[method, p]
                n, w, ci = r["shots"], r["logical_failures"], r["wilson_95"]
                source = path.relative_to(ROOT).as_posix()
                parts = r["layouts"]
            else:
                parts = [read(f) for f in sorted((BATH / "checkpoints").glob(f"{method}-p{p:g}-layout*.json"))]
                n = sum(r["shots"] for r in parts)
                w = sum(r["logical_failures"] for r in parts)
                ci = None
                source = "experiments/million-bath-03/checkpoints"
            if p == .001:
                native_parts[method] = parts
            chunks = [chunk for part in parts for chunk in part["chunks"]]
            if sum(c["shots"] for c in chunks) != n:
                raise ValueError("Bath timing denominator differs from scored shots")
            ns = sum(c["decode_ns_under_campaign_load"] for c in chunks)
            rows.append(row("bath", 7, "Z", p, method, TARGET, n, w, source,
                            ns=ns, rounds_sum=n*7, archived=timing[method, p], saved_ci=ci))
    timing = {r["arm"]: r for r in read(EXP / "bath-clock-controls-02/timing.json")["arms"]}
    path = CONTROLS / "result.json"
    published = {r["arm"]: r for r in read(path)["arms"]} if path.exists() else {}
    manifest = CONTROLS / "manifest.json"
    frozen = read(manifest)["layout_metadata_by_arm"] if manifest.exists() else {}
    warm_path = CONTROLS / "accelerated-timing.json"
    warmed = {r["arm"]: r for r in read(warm_path)["arms"]} if warm_path.exists() else {}
    for method in POLICIES:
        ci = None
        if method in published:
            r = published[method]
            n, w, ci = r["shots"], r["logical_errors"], r.get("conditional_binomial_clopper_pearson_95")
            source = path.relative_to(ROOT).as_posix()
        elif method in ("fixed_6", "individual", "fft_timing") and method in frozen:
            native_arm = "individual" if method == "fft_timing" else method
            parts = native_parts[native_arm]
            for layout in parts:
                control = frozen[method][layout["layout_index"] - 1]
                fields = (("physical_circuit_sha256", "physical_exact_circuit_sha256"),
                          ("prior_circuit_sha256", "prior_exact_circuit_sha256"),
                          ("bath_ids_in_data_qubit_order", "bath_ids"),
                          ("herald_mask_sha256", "herald_mask_sha256"),
                          ("layout_seed", "layout_seed"))
                if any(layout[a] != control[b] for a,b in fields):
                    raise ValueError(f"Shared control fingerprints differ for {method}, layout {layout['layout_index']}")
            n = sum(r["shots"] for r in parts)
            w = sum(r["logical_failures"] for r in parts)
            source = f"experiments/million-bath-03 (shared {native_arm}/0.001)"
        else:
            folder = "chunks"
            if method == "fft_only" and (CONTROLS / "paired-chunks" / method).exists():
                folder = "paired-chunks"
            parts = [read(f) for f in sorted((CONTROLS / folder / method).glob("layout-*/chunk-*.json"))]
            n = sum(r["shots"] for r in parts)
            w = sum(r["logical_errors"] for r in parts)
            source = f"experiments/million-controls-03/{folder}/{method}"
        paired = published.get(method, {}).get("paired_errors_vs_individual", {})
        rows.append(row("controls", 7, "Z", .001, method, TARGET, n, w, source,
                        archived=timing[method], warmed=warmed.get(method), exact=True, saved_ci=ci,
                        recovered=paired.get("fitted_only_fails"), introduced=paired.get("fft_only_fails")))
    return rows


def pct(value, decimals=3):
    return f"{100*value:.{decimals}f}"


def fmt_us(x):
    if x is None:
        return "-"
    return f"{x:,.3f}" if x < 10 else f"{x:,.2f}" if x < 100 else f"{x:,.1f}"


class Sheet:
    def __init__(self, path, stamp, status, footer="One-million-shot studies"):
        self.c = canvas.Canvas(str(path), pagesize=letter, pageCompression=1)
        self.c.setTitle("Logical error and decoder runtime | One-million-shot studies")
        self.c.setAuthor("MidQWest decoder study")
        self.stamp, self.status, self.footer, self.number = stamp, status, footer, 0
        self.ink, self.muted, self.teal = "#132C3A", "#435F6A", "#006D6C"

    def text(self, x, y, text, size=9, font="Helvetica", color=None, right=False):
        self.c.setFillColor(colors.HexColor(color or self.ink))
        self.c.setFont(font, size)
        (self.c.drawRightString if right else self.c.drawString)(x, y, str(text))

    def rule(self, y, color="#C7D6D7"):
        self.c.setStrokeColor(colors.HexColor(color))
        self.c.setLineWidth(.6)
        self.c.line(34, y, 578, y)

    def page(self, section, title, description):
        if self.number:
            self.c.showPage()
        self.number += 1
        self.text(34, 758, "MIDQWEST  /  DECODER STUDY", 9, "Helvetica-Bold", self.teal)
        self.text(578, 758, self.status, 8.6, "Helvetica-Bold", self.teal, True)
        self.rule(745)
        self.text(34, 719, section.upper(), 9, "Helvetica-Bold", self.muted)
        self.text(34, 683, title, 25, "Helvetica-Bold")
        self.text(34, 659, description, 10, color=self.muted)
        self.rule(45)
        self.text(34, 29, self.stamp, 8, color=self.muted)
        self.text(578, 29, f"{self.footer}   /   {self.number}", 8, color=self.muted, right=True)
        self.y = 630

    def paragraph(self, text):
        words, line = text.split(), ""
        for word in words:
            test = (line + " " + word).strip()
            if pdfmetrics.stringWidth(test, "Helvetica", 9) > 537:
                self.text(34, self.y, line, 9, color=self.muted)
                self.y -= 13
                line = word
            else:
                line = test
        if line:
            self.text(34, self.y, line, 9, color=self.muted)
            self.y -= 13
        self.y -= 6
        if self.y < 45:
            raise ValueError(f"Content exceeds page {self.number}")

    def table(self, title, rows, label, label_header="Condition / decoder", compare_timing=False):
        self.text(34, self.y, title, 12, "Helvetica-Bold")
        self.y -= 20
        self.c.setFillColor(colors.HexColor("#E7F1F0"))
        self.c.rect(34, self.y-5, 544, 17, fill=1, stroke=0)
        positions = [38, 237, 296, 355, 456, 516, 574]
        headers = [label_header, "Shots", "Wrong", "LER %", "95% interval %", "µs / shot", "µs / round"]
        if compare_timing:
            headers[-2:] = ["New \u00b5s / shot", "Old \u00b5s / shot"]
        for i, (x, item) in enumerate(zip(positions, headers)):
            self.text(x, self.y, item, 8, "Helvetica-Bold", right=bool(i))
        self.y -= 19
        for i, r in enumerate(rows):
            n = r["shots"]
            if i % 2:
                self.c.setFillColor(colors.HexColor("#F4F7F7"))
                self.c.rect(34, self.y-5, 544, 18, fill=1, stroke=0)
            fields = [label(r), f"{n:,}", f'{r["wrong"]:,}' if n else "-",
                      pct(r["ler"], 4) if n else "-",
                      f'{pct(r["lower_95"],4)} - {pct(r["upper_95"],4)}' if n else "-",
                      fmt_us(r["us_per_shot"]), fmt_us(r["us_per_round"])]
            if compare_timing:
                fields[-1] = fmt_us(r["archived_us_per_shot"])
            for j, (x, field) in enumerate(zip(positions, fields)):
                self.text(x, self.y, field, 8.1 if j != 4 else 7.8,
                          "Helvetica-Bold" if j == 3 and n else "Helvetica", right=bool(j))
                if j > 1 and pdfmetrics.stringWidth(field, "Helvetica", 8.1) >= positions[j] - positions[j-1] - 2:
                    raise ValueError(f"Column too narrow: {field}")
            self.y -= 18
        self.y -= 8
        if self.y < 45:
            raise ValueError(f"Table exceeds page {self.number}")


def draw_bath(s, bath, controls):
    finished = all(r["shots"] == r["target_shots"] for r in bath + controls)
    subtitle = ("All 18 rows score 1,000,000 shots on five fixed layouts." if finished
                else "The planned sample is 1,000,000 shots per schedule and level.")
    s.page("Dual-rail bath scheduling", "Logical error and decoder runtime",
           f"Distance 7, Z memory, seven rounds. {subtitle}")
    s.table("Fixed clock and per-qubit timing", bath,
            lambda r: f'{100*r["probability"]:g}%   {r["method_label"]}', "Added X/Z error / schedule", compare_timing=True)
    s.paragraph("Each round adds the stated error per data-qubit axis. Reset and readout retain 0.1% flips. New sweep runtimes time accelerated decoding on 10,000-shot campaign batches under concurrent load. Archived author timings used 4,000-shot batches on a serial CPU.")
    s.table("Calibration and schedule controls at 0.1% added error", controls, lambda r:r["method_label"], "Schedule", compare_timing=True)
    fft = next(r for r in controls if r["method"] == "fft_only")
    shared = "The 0.1% clock and fitted controls share sweep records. FFT timing with the fitted frame uses the same fitted records."
    if fft["recovered"] is not None:
        shared += (f' On {fft["shots"]:,} matched shots, FFT-only weights corrected {fft["recovered"]:,} fitted failures'
                   f' and introduced {fft["introduced"]:,}.')
    s.paragraph(shared)
    interval_note = "The sweep uses Wilson intervals and the controls use Clopper-Pearson intervals, conditional on the five fixed layouts."
    if not finished:
        interval_note += " Partial rows reflect the layouts already scored."
    s.paragraph(interval_note)
    s.paragraph("New control timings average five layout medians from warmed 512-shot decoder calls under shared load, five timed calls per layout. Archived controls used the author decoder on 512-shot batches. Divide each value by seven for time per syndrome round.")


def generate(path, rows, stamp, complete):
    s = Sheet(path, stamp, "COMPLETE" if complete else "PROGRESS SNAPSHOT")
    def group(name):
        return [r for r in rows if r["group"] == name]
    def condition(r):
        return f'{r["basis"]}  {100*r["probability"]:g}%   {r["method_label"]}'
    s.page("Campaign overview", "One-million-shot decoder studies",
           "A logical failure is a shot where the predicted logical observable differs from Stim's recorded value.")
    sweep = group("bath")
    s.table("Dual-rail checks at the lowest and highest backgrounds",
            [r for r in sweep if r["probability"] in (.001, .04)],
            lambda r:f'{100*r["probability"]:g}%   {r["method_label"]}', "Added X/Z error / schedule")
    s.paragraph("Each schedule selects a check time and correction frame for a simulated dual-rail qubit. The sweep holds the five bath layouts fixed and changes a common background error. Bath times here measure the accelerated decoder on the scored campaign batches, excluding sampling.")
    s.table("Fixed-circuit decoder comparison at 0.5% noise",
            [r for r in group("grid") if r["basis"] == "Z" and r["probability"] == .005],
            lambda r:f'd{r["distance"]}    {r["method_label"]}', "Distance / decoder")
    s.paragraph("These rows change the decoder while keeping each circuit's detector samples identical. They use live campaign batch measurements. The physical noise and timing protocols differ from the dual-rail study.")
    valid = [r for r in rows if r["group"] != "pooled"]
    done = sum(r["shots"] == r["target_shots"] for r in valid)
    live = sum(0 < r["shots"] < r["target_shots"] for r in valid)
    empty = len(valid) - done - live
    s.text(34, s.y, "Measurement coverage", 12, "Helvetica-Bold")
    s.y -= 20
    s.paragraph(f"{done} of {len(valid)} reported rows have reached their sample targets. {live} rows have partial results and {empty} await their first verified chunk. Each row reports exact scored shots. Decoders and equivalent timing policies can share records.")
    s.paragraph("Pages 2 and 3 cover bath schedules, prior policies and paired fault removals. Pages 4 through 7 give the fixed-circuit sweep, and pages 8 and 9 cover scaling through distance 31. The accompanying CSV includes every pooled stratum and raw fixed-circuit batch timings.")
    s.paragraph("The FFT frequency calibration uses 49 modeled bath traces. The post-catch pulse-hold study computes physical loss probabilities at modeled endpoints. Neither quantity has a million-shot logical-error denominator.")
    draw_bath(s, group("bath"), group("controls"))
    s.page("Fixed-circuit comparisons", "Decoder policies and fault sources",
           "The pooled policy comparison targets 1,000,000 total shots per decoder across 24 circuit settings.")
    s.table("Pooled decoder policies", group("pooled_total"), lambda r:r["method_label"], "Decoder")
    s.paragraph("The 24 settings cross distances 3, 5 and 7, X and Z memory, and noise levels 0.1%, 0.3%, 0.5% and 1%. The basis-selected prior uses clock weights on X and equal-axis weights on Z, with timing from those calls. The pooled Wilson interval approximates uncertainty for the observed mix. Each stratum targets 41,666 or 41,667 shots.")
    s.table("Fault-family removal at distance 7, Z memory", group("fault"),
            lambda r:f'{100*r["probability"]:g}%   {r["method_label"]}', "Circuit noise / intervention")
    s.paragraph("Each fault row targets 1,000,000 shots. The full sample combines four independent fault streams. Each removal subtracts one stream on the same records and uses the full-noise correlated matcher. Removal effects can overlap.")
    s.paragraph("Fixed-circuit intervals use Wilson 95% limits. Timings throughout the fixed-circuit tables divide summed batch decoding wall time by the observed shots or syndrome rounds. Parallel jobs shared the CPU.")
    for d in (3, 5, 7):
        for extended in ([False, True] if d == 7 else [False]):
            records = [r for r in group("grid") if r["distance"] == d and (r["probability"] > .01) == extended]
            s.page("Fixed-circuit noise sweep", f"Distance {d} at {'higher' if extended else 'standard'} noise",
                   f"Rotated X and Z memory with {d} syndrome rounds. Each decoder and circuit targets 1,000,000 shots.")
            for b in ("X","Z"):
                s.table(f"{b} memory", [r for r in records if r["basis"] == b], condition, "Basis / noise / decoder")
            s.paragraph("Ordinary and correlated MWPM decode identical Stim detector records at each circuit setting. Circuit noise covers gate depolarization, round-start data depolarization, pre-measurement flips and post-reset flips.")
            if not extended:
                s.paragraph("The 0.5% Z-memory comparison includes the prespecified posterior lookup at distance 3 and selective correlated matching at distances 5 and 7.")
            s.paragraph("Intervals are Wilson 95% limits. Runtime comes from live campaign batch calls under concurrent load, including unpacking performed by the decoder wrapper. Sampling and compilation are excluded.")
    for p in (.005,.01):
        s.page("Code-distance scaling", f"Scaling at {100*p:g}% circuit noise",
               "Each distance, basis and decoder targets 1,000,000 shots. Rows report their own observed counts.")
        vals = [r for r in group("large") if r["probability"] == p]
        for d in sorted({r["distance"] for r in vals}):
            s.table(f"Distance {d}", [r for r in vals if r["distance"] == d],
                    lambda r:f'{r["basis"]}    {r["method_label"]}', "Basis / decoder")
        s.paragraph("Matching variants use either the native detector error model, an equal-axis expansion, or a clock-derived prior. Each decoder replays identical seeded Stim batches at a given distance, basis and noise level.")
        s.paragraph("Wilson 95% intervals describe shot uncertainty. The timing columns use measured batch wall time for completed chunks under concurrent load. Individual decoder jobs can reach different sample sizes during collection.")
    s.c.save()


def publish_bath(destination, rows):
    bath = [r for r in rows if r["group"] == "bath"]
    controls = [r for r in rows if r["group"] == "controls"]
    if ((BATH / "result.json").exists() and (CONTROLS / "result.json").exists()
            and all(r["shots"] == TARGET and r["us_per_shot"] is not None for r in bath + controls)):
        inputs = [BATH / "result.json", CONTROLS / "result.json", CONTROLS / "accelerated-timing.json"]
        if destination.exists() and destination.stat().st_mtime >= max(
                path.stat().st_mtime for path in [Path(__file__), *inputs]):
            return True
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Use the publication time of the latest scientific input so fixed results have a stable timestamp.
        latest = datetime.fromtimestamp(max(path.stat().st_mtime for path in inputs), timezone.utc)
        bath_stamp = latest.strftime("Final records available %Y-%m-%d %H:%M:%S UTC")
        with NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
        try:
            sheet = Sheet(tmp_path, bath_stamp, "COMPLETE \u00b7 BATH STUDY", footer="Million-shot bath study")
            draw_bath(sheet, bath, controls)
            sheet.c.save()
            tmp_path.replace(destination)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(f"Published 18-row bath sheet: {destination}", file=sys.stderr)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--bath-pdf", type=Path, default=BATH_PDF,
                        help="Standalone completed bath sheet, emitted once both final study records exist")
    parser.add_argument("--bath-only", action="store_true", help="Build only the completed standalone bath sheet")
    parser.add_argument("--require-complete", action="store_true", help="Refuse to emit a progress snapshot")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("Collection began %Y-%m-%d %H:%M:%S UTC")
    rows = bath_rows()
    if args.bath_only:
        if not publish_bath(args.bath_pdf, rows):
            parser.exit(2, "Bath accuracy and timing records are still being completed.\n")
        return
    rows.extend(fixed_rows())
    complete = all(r["shots"] == r["target_shots"] and r["us_per_shot"] is not None for r in rows)
    if args.require_complete and not complete:
        parser.exit(2, "At least one accuracy or timing row is incomplete.\n")
    if complete:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, str(FIXED / "audit.py"), "--require-complete", "--replay-lookup"],
                       cwd=ROOT, env=env, stdout=sys.stderr, check=True)
    for path in (args.pdf, args.csv):
        path.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(delete=False, dir=args.pdf.parent, suffix=".tmp") as tmp:
        pdf_tmp = Path(tmp.name)
    try:
        generate(pdf_tmp, rows, stamp, complete)
        pdf_tmp.replace(args.pdf)
    finally:
        pdf_tmp.unlink(missing_ok=True)
    with NamedTemporaryFile(mode="w", delete=False, newline="", encoding="utf-8", dir=args.csv.parent, suffix=".tmp") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["snapshot_utc", *rows[0]])
        writer.writeheader()
        writer.writerows({"snapshot_utc": stamp.split("began ")[1], **r} for r in rows)
        csv_tmp = Path(tmp.name)
    csv_tmp.replace(args.csv)
    publish_bath(args.bath_pdf, rows)
    if complete:
        print("Complete")
    else:
        print(f"Progress snapshot: {len(rows)} CSV rows | {args.pdf}")


if __name__ == "__main__":
    main()
