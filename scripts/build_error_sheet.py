"""Build the one-page MidQWest error and runtime sheet from the archived measurements."""

import csv
import json
import math
from pathlib import Path

from matplotlib import get_data_path
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUT = ROOT / "output/pdf/Decoder error rates.pdf"
WIDTH, HEIGHT = 1280, 925
L, R, COL = 43, 660, 577
INK, MUTED, TEAL, RULE = "#1b333c", "#46616d", "#006c61", "#c5d3d4"


def data(path):
    return json.loads((EXPERIMENTS / path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with (EXPERIMENTS / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def error(n, shots, places=3):
    if not 0 <= n <= shots or shots <= 0:
        raise ValueError("Invalid logical error count")
    return f"{100 * n / shots:.{places}f}%"


def number(value, places):
    if value <= 0 or not math.isfinite(value):
        raise ValueError("Expected positive measured batch time")
    return f"{value:,.{places}f}"


class Sheet:
    def __init__(self, canvas):
        self.canvas = canvas

    def text(self, x, y, value, *, size=11.5, bold=False, color=INK, end=False):
        self.canvas.setFont("SheetBold" if bold else "SheetRegular", size)
        self.canvas.setFillColor(HexColor(color))
        fn = self.canvas.drawRightString if end else self.canvas.drawString
        fn(x, HEIGHT - y, value)

    def line(self, x, y, length=COL):
        self.canvas.setStrokeColor(HexColor(RULE))
        self.canvas.setLineWidth(.65)
        self.canvas.line(x, HEIGHT - y, x + length, HEIGHT - y)

    def section(self, x, y, title, subtitle):
        self.line(x, y - 22)
        self.text(x, y, title, size=18, bold=True)
        self.text(x, y + 18, subtitle, size=10, color=MUTED)

    def cells(self, x, y, cells, *, header=False, highlight=False, size=11):
        for offset, value, right in cells:
            self.text(x + offset, y, value, size=size, end=right, bold=header or highlight,
                      color=MUTED if header else TEAL if highlight else INK)

    def table_header(self, x, y, name="Decoder"):
        self.cells(x, y, [(0, name, False), (266, "Logical error", True),
                          (402, "Wrong / shots", True), (496, "µs/shot", True),
                          (COL, "µs/round", True)], header=True, size=10.5)

    def row(self, x, y, label, n, shots, batch, per_round, *, precision=3, time_places=3, highlight=False, label_offset=0):
        self.cells(x, y, [(label_offset, label, False), (266, error(n, shots, precision), True),
                          (402, f"{n:,} / {shots:,}", True), (496, number(batch, time_places), True),
                          (COL, number(per_round, time_places), True)], highlight=highlight)


def reference(sheet):
    sheet.section(L, 117, "Decoder comparison by distance", "Z memory  ·  0.5% circuit noise  ·  rounds equal distance")
    sheet.table_header(L, 156)
    rows = csv_rows("circuit-frontier-01/metrics.csv")
    for group, distance in enumerate((3, 5, 7)):
        y = 178 + 67 * group
        case = f"{'primary' if distance == 3 else 'confirmation'}-d{distance}-z-p0.005-ordinary"
        by_method = {r["method"]: r for r in rows if r["case"] == case}
        choice = ("fast_hybrid", "Posterior lookup") if distance == 3 else ("css_routed", "Selective correlated matching")
        sheet.text(L, y, f"Distance {distance}", bold=True, size=11.5)
        for i, (method, label) in enumerate((("mwpm", "Ordinary PyMatching"), choice, ("correlated", "Always correlated matching"))):
            row = by_method[method]
            n, shots = int(row["errors"]), int(row["shots"])
            if abs(100 * n / shots - float(row["ler_percent"])) > 1e-9:
                raise ValueError(f"Rate mismatch for {case}/{method}")
            sheet.row(L, y + 17 + 15 * i, label, n, shots, float(row["batch_us_shot"]),
                      float(row["batch_us_round"]), precision=4, highlight=method == choice[0])


def pooled(sheet):
    sheet.section(L, 404, "Held-out decoder benchmark", "24 scenarios  ·  distances 3, 5 and 7  ·  X and Z memories  ·  480,000 shots per decoder")
    sheet.table_header(L, 443)
    results = data("iteration-04/accuracy/summary.json")
    labels = [("native", "Standard correlated matching"), ("bath_clock", "Bath-weighted matching"),
              ("belief", "Belief matching"), ("frozen", "Distance-based hybrid")]
    for i, (key, label) in enumerate(labels):
        r = results[key]
        sheet.row(L, 463 + i * 18, label, r["failures"], r["shots"], r["batch_us_per_shot"],
                  r["batch_us_per_shot"] / 5, precision=4, time_places=2, highlight=key == "frozen")

    sheet.section(L, 567, "Bath weights on fixed detector records", "Separate 24-scenario confirmation  ·  3,000,000 identical shots per policy")
    sheet.table_header(L, 607, "Correlated-matching policy")
    result = data("bath-basis-policy-01/confirmation/summary.json")["pooled"]
    for i, (key, label) in enumerate((("equal", "Always equal-axis weights"), ("bath", "Always bath weights"),
                                      ("policy", "Bath weights only in X memory"))):
        r = result["methods"][key]
        sheet.row(L, 626 + i * 18, label, r["failures"], result["shots"], r["batch_us_per_shot"],
                  r["batch_us_per_shot"] / 5, precision=4, time_places=2, highlight=key == "policy")


def source_removals(sheet):
    sheet.section(L, 710, "Which faults drive logical failures?", "Distance 7  ·  Z memory  ·  correlated matching  ·  20,000 shots per column")
    sheet.cells(L, 752, [(0, "Fault family removed", False), (302, "0.5% noise", True),
                         (435, "1% noise", True), (COL, "2% noise", True)], header=True)
    rows = csv_rows("high-noise-categorization-01/fault-sources/removals.csv")
    lookup = {(float(r["p"]), r["removed"]): r for r in rows}
    for i, (key, label) in enumerate((("all", "None"), ("gate", "Gate faults"), ("round_start_data", "Round-start data faults"),
                                      ("readout", "Readout flips"), ("reset", "Reset flips"))):
        cells = [(0, label, False)]
        for offset, p in zip((302, 435, COL), (.005, .01, .02), strict=True):
            row = lookup[p, "gate" if key == "all" else key]
            n, shots = int(row["full_failures" if key == "all" else "remaining_failures"]), int(row["shots"])
            cells.append((offset, f"{error(n, shots)}  ({n:,})", True))
        sheet.cells(L, 773 + i * 18, cells)
    sheet.text(L, 866, "Parentheses give wrong shots. Removing different faults can rescue the same shot.", size=9, color=MUTED)


def bath(sheet):
    sheet.section(R, 117, "Sakidja bath timing under higher noise", "Distance 7  ·  Z memory  ·  7 rounds  ·  same five layouts per check schedule")
    sheet.table_header(R, 156, "Background / check schedule")
    run = data("bath-background-stress-01/result.json")
    timing = data("bath-background-stress-01/timing/result.json")
    times = {(r["p_check"], r["arm"]): r for r in timing["points"]}
    schedule = {a["arm"]: i for i, a in enumerate(run["arms"])}
    levels = sorted({r["p_check"] for r in run["points"]})
    for row in run["points"]:
        p, arm = row["p_check"], row["arm"]
        t = times[p, arm]
        if (t["logical_failures"], t["shots"]) != (row["logical_failures"], row["shots"]):
            raise ValueError("Bath timing replay and error counts disagree")
        i = schedule[arm]
        y = 180 + 40 * levels.index(p) + 16 * i
        if i == 0:
            sheet.text(R, y, f"{p * 100:g}%", size=11)
        sheet.row(R, y, ("Fixed clock", "Bath-timed check")[i], row["logical_failures"], row["shots"],
                  t["batch_us_shot"], t["batch_us_round"], time_places=2, highlight=bool(i), label_offset=46)


def stress(sheet):
    sheet.section(R, 404, "Fixed-circuit noise stress test", "Distance 7  ·  7 rounds  ·  20,000 shots per memory basis at each noise level")
    sheet.cells(R, 440, [(210, "X memory", False), (441, "Z memory", False)],
                header=True)
    sheet.line(R + 126, 446, 218)
    sheet.line(R + 358, 446, COL - 358)
    sheet.cells(R, 462, [(0, "Noise / decoder", False), (240, "Error (wrong)", True), (344, "µs/shot · /round", True),
                         (468, "Error (wrong)", True), (COL, "µs/shot · /round", True)], header=True, size=9.8)
    index = {(float(r["p"]), r["basis"]): r for r in csv_rows("high-noise-categorization-01/fixed-stim/scenarios.csv")}
    for g, p in enumerate(sorted({p for p, _ in index})):
        for m, method in enumerate(("ordinary", "correlated")):
            cells = [(44, method.capitalize(), False)]
            if m == 0:
                cells.append((0, f"{100*p:g}%", False))
            for basis, (er, tm) in (("x", (240, 344)), ("z", (468, COL))):
                r = index[p, basis]
                n, shots = int(r[method + "_errors"]), int(r["shots"])
                t = float(r[method + "_batch_us_per_shot"])
                cells.extend([(er, f"{error(n, shots)} ({n:,})", True),
                              (tm, f"{t:.2f} · {t / int(r['rounds']):.2f}", True)])
            sheet.cells(R, 486 + 39 * g + 16 * m, cells, size=10.2)


def scaling(sheet):
    sheet.section(R, 710, "Large-code stress test", "Distance 31  ·  Z memory  ·  1% circuit noise  ·  31 rounds  ·  961 data qubits")
    sheet.table_header(R, 752)
    rows = {r["method"]: r for r in csv_rows("qubit-scaling-01/high-noise/measured/metrics.csv")
            if r["distance"] == "31" and r["status"] == "ok"}
    labels = (("mwpm", "Ordinary PyMatching"), ("correlated", "Standard correlated matching"),
              ("expanded_corr", "Equal-axis correlated matching"), ("clock_corr", "Bath-weighted matching"),
              ("union_find", "Weighted Union-Find"), ("css_routed", "Selective correlated matching"))
    for i, (k, label) in enumerate(labels):
        r = rows[k]
        t = float(r["batch_us_per_shot"])
        sheet.row(R, 772 + i * 17, label, int(r["errors"]), int(r["shots"]), t, t / 31,
                  precision=2, time_places=2)


def main():
    font_dir = Path(get_data_path()) / "fonts/ttf"
    for name, file in (("SheetRegular", "DejaVuSans.ttf"), ("SheetBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    canvas = Canvas(str(OUTPUT), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    canvas.setTitle("Logical error and decoder runtime | MidQWest 2026")
    sheet = Sheet(canvas)
    sheet.text(L, 48, "Logical error and decoder runtime", bold=True, size=32)
    sheet.text(L, 73, "MidQWest 2026   ·   Surface-code memory experiments   ·   Logical failures per 100 complete shots",
               size=14, color=MUTED)
    reference(sheet)
    pooled(sheet)
    source_removals(sheet)
    bath(sheet)
    stress(sheet)
    scaling(sheet)
    sheet.line(L, 879, WIDTH - 2 * L)
    sheet.text(L, 897, "Batch times cover decoder inference. µs/round divides batch time by all syndrome rounds. Batch sizes and timing protocols vary by study.",
               size=10, color=MUTED)
    sheet.text(L, 913, "Fixed circuits use the provisional simple_uniform noise model. The bath study varies background X and Z errors per qubit per round and compares physical check schedules.",
               size=10, color=MUTED)
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    main()
