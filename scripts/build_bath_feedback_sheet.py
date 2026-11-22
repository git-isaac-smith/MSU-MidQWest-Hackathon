"""Build Sakidja's revised one-page logical-error and runtime sheet from measured records."""

from pathlib import Path

from matplotlib import get_data_path
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from build_error_sheet import ROOT, EXPERIMENTS, csv_rows, data


W, H = 1248, 1016
LEFT, RIGHT, COL = 40, 648, 560
INK, MUTED, TEAL, LIGHT, RULE = "#142f38", "#415d69", "#006d61", "#edf6f3", "#c2d4d2"
OUTPUT = ROOT / "output/pdf/Decoder error rates - bath timing.pdf"


class Page:
    def __init__(self, canvas):
        self.c = canvas

    def text(self, x, y, message, size=12, *, bold=False, color=INK, right=False):
        self.c.setFillColor(HexColor(color))
        self.c.setFont("SheetBold" if bold else "SheetRegular", size)
        (self.c.drawRightString if right else self.c.drawString)(x, H-y, str(message))

    def line(self, x, y, width):
        self.c.setStrokeColor(HexColor(RULE))
        self.c.setLineWidth(.7)
        self.c.line(x, H-y, x+width, H-y)

    def row(self, x, y, values, columns, *, header=False, accent=False, size=11.4):
        for j, (s, offset) in enumerate(zip(values, columns, strict=True)):
            self.text(x+offset, y, s, size, bold=header or accent, color=MUTED if header else TEAL if accent else INK,
                      right=j > 0)

    def section(self, x, y, title, subtitle):
        self.line(x, y-23, COL)
        self.text(x, y, title, 18, bold=True)
        self.text(x, y+19, subtitle, 10.6, color=MUTED)


def get_sources():
    rare = data("bath-rare-confirmation-02/result.json")
    stress = data("bath-background-stress-01/result.json")
    timings = data("bath-background-stress-01/timing/result.json")
    times = {(r["arm"], r["p_check"]): r for r in timings["points"]}
    fixed_name, timed_name = [r["arm"] for r in stress["arms"]]
    by_level = {}
    for r in stress["points"]:
        t = times[r["arm"], r["p_check"]]
        assert (t["shots"], t["logical_failures"]) == (r["shots"], r["logical_failures"])
        by_level.setdefault(r["p_check"], {})[r["arm"]] = (r, t["batch_us_shot"])
    controls = data("bath-clock-controls-02/result.json")["arms"] + data("bath-clock-controls-02/fft_result.json")["arms"]
    return rare, by_level, fixed_name, timed_name, controls


def main():
    fonts = Path(get_data_path()) / "fonts/ttf"
    for name, filename in (("SheetRegular", "DejaVuSans.ttf"), ("SheetBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    rare, levels, fixed, timed, controls = get_sources()
    fft = data("bath-fft-timing-02/summary.json")
    noisy_fft = data("bath-fft-timing-02/sensitivity.json")["results"]["10000"]
    pulse = data("bath-pulse-hold-02/measured/summary.json")
    hybrid = data("iteration-04/accuracy/summary.json")
    rate = lambda n, total, prec: f"{100*n/total:.{prec}f}%"
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    c = Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("Logical error and decoder runtime | Bath-timed checks")
    p = Page(c)
    p.text(LEFT, 31, "MidQWest 2026     /     Northrop Grumman decoder challenge", 12, bold=True, color=TEAL)
    p.text(LEFT, 72, "Logical error and decoder runtime", 34, bold=True)
    p.text(LEFT, 100, "Ridwan Sakidja’s bath map tells each dual-rail qubit when to check for the returning excitation.", 16, color=MUTED)
    c.setFillColor(HexColor(LIGHT))
    c.roundRect(LEFT, H-205, W-2*LEFT, 84, 7, stroke=0, fill=1)
    for x, title in ((60, "Fixed clock at the original noise"), (450, "Per-qubit timing on fresh shots"), (845, "At the heaviest background noise")):
        p.text(x, 143, title, 11, bold=True, color=MUTED)
    base = levels[min(levels)][fixed][0]
    p.text(60, 174, rate(base["logical_failures"], base["shots"], 3), 25, bold=True)
    p.text(60, 192, f'{base["logical_failures"]:,} wrong / {base["shots"]:,} shots', 11)
    p.text(450, 174, rate(rare["logical_failures"], rare["shots"], 4), 25, bold=True, color=TEAL)
    ci = [100*i for i in rare["wilson_95"]]
    p.text(450, 192, f'{rare["logical_failures"]:,} / {rare["shots"]:,}  ·  95% CI {ci[0]:.4f}–{ci[1]:.4f}%', 10.3)
    high = levels[max(levels)]
    ratio = high[fixed][0]["logical_error_rate"]/high[timed][0]["logical_error_rate"]
    p.text(845, 174, f'{ratio:.2f}× lower error', 23, bold=True, color=TEAL)
    p.text(845, 192, f'{100*high[fixed][0]["logical_error_rate"]:.3f}% fixed clock vs. {100*high[timed][0]["logical_error_rate"]:.3f}% timed', 10.7)

    p.section(LEFT, 252, "Bath timing across five noise levels", "Distance 7 · Z memory · 7 rounds · 20,000 shots per schedule and level")
    cols = (0, 211, 282, 477, 560)
    p.row(LEFT, 298, ("Added X/Z", "Fixed clock % (wrong)", "µs/shot", "Bath-timed % (wrong)", "µs/shot"), cols, header=True, size=10.2)
    for i, (background, pair) in enumerate(sorted(levels.items())):
        cr, ct = pair[fixed]
        br, bt = pair[timed]
        p.row(LEFT, 321+20*i, (f"{100*background:g}%", f'{rate(cr["logical_failures"],cr["shots"],3)} ({cr["logical_failures"]:,})',
              f"{ct:,.1f}", f'{rate(br["logical_failures"],br["shots"],3)} ({br["logical_failures"]:,})', f"{bt:,.1f}"), cols)
    p.text(LEFT, 423, "Fitted check and frame; both decoders receive loss locations and score every shot.", 10, color=MUTED)

    p.section(LEFT, 469, "Does each qubit need its own time?", "Five layouts · 20,000 fresh shots per policy · 0.1% added X/Z error per axis")
    control_times = {a["arm"]: a for a in data("bath-clock-controls-02/timing.json")["arms"]}
    labels = {"fixed_6":"Fixed clock", "global":"One time from training baths",
              "global_installed_calibration":"One time from installed baths", "shuffled":"Same times, shuffled among qubits",
              "individual":"Each qubit’s fitted time", "fft_timing":"Full FFT time with fitted frame",
              "dominant_tone_timing":"Single-tone time with fitted frame", "fft_only":"FFT time, frame and decoder weights"}
    control_cols = (0,270,350,412,492,560)
    p.row(LEFT, 511, ("Schedule","Mean time¹","Error","Wrong","µs/shot","µs/round"), control_cols, header=True, size=10.1)
    for i, a in enumerate(controls):
        key = a["arm"]
        timing = control_times[key]
        p.row(LEFT, 532+17*i, (labels[key], f'{a["time_distribution"]["mean"]:.2f}',
                  rate(a["logical_errors"],a["shots"],3), f'{a["logical_errors"]:,}',
                  f'{timing["batch_us_shot"]:,.1f}',f'{timing["batch_us_round"]:,.1f}'),
              control_cols, accent=key=="individual", size=10.7)
    schedules = {a["arm"]: a for a in controls}
    p.text(LEFT, 675, f'Identical waiting times, reassigned among qubits: {100*schedules["shuffled"]["ler"]:.3f}% logical error',
           11, bold=True, color=TEAL)
    p.text(LEFT, 690, f'vs. {100*schedules["individual"]["ler"]:.3f}% when each qubit receives its own fitted time.', 10.8, color=MUTED)

    p.section(RIGHT, 252, "Decoder accuracy on fixed Stim shots", "Z memory · 0.5% circuit noise · each distance uses the same records across decoders")
    ref = csv_rows("circuit-frontier-01/metrics.csv")
    cols = (0, 289, 417, 491, 560)
    p.row(RIGHT, 298, ("Distance / decoder","Logical error","Wrong / shots","µs/shot","µs/round"), cols, header=True, size=10.4)
    for g, d in enumerate((3,5,7)):
        name = f'{"primary" if d==3 else "confirmation"}-d{d}-z-p0.005-ordinary'
        group = {r["method"]:r for r in ref if r["case"]==name}
        candidate = "fast_hybrid" if d==3 else "css_routed"
        for i, (key,label) in enumerate((("mwpm","Ordinary PyMatching"),(candidate,"Posterior lookup" if d==3 else "Selective correlated matching"),("correlated","Always correlated matching"))):
            r=group[key]
            prefix=f"d{d}   " if i==0 else "       "
            p.row(RIGHT, 320+57*g+17*i, (prefix+label, rate(int(r["errors"]),int(r["shots"]),4),
                f'{int(r["errors"]):,} / {int(r["shots"]):,}',f'{float(r["batch_us_shot"]):.3f}',f'{float(r["batch_us_round"]):.3f}'),
                cols, accent=key==candidate, size=10.8)
    p.text(RIGHT, 494, "Lookup is 12.4× faster at d3. At d5 and d7, selective matching improves", 11, color=MUTED)
    p.text(RIGHT, 511, "on ordinary matching at similar or better speed. Always-correlated decoding", 11, color=MUTED)
    p.text(RIGHT, 528, "has the fewest errors there and costs 2.9–3.4× as much as selective matching.", 11, color=MUTED)

    p.section(RIGHT, 574, "The hybrid trades speed for accuracy", "480,000 shots per decoder · 24 fixed-circuit scenarios · distances 3, 5 and 7")
    p.row(RIGHT, 617, ("Decoder","Logical error","Wrong / shots","µs/shot","µs/round"), cols, header=True, size=10.4)
    for i,(k,label) in enumerate((("native","Standard correlated matching"),("frozen","Distance-based hybrid"))):
        r=hybrid[k]
        p.row(RIGHT, 636+18*i,(label,rate(r["failures"],r["shots"],4),f'{r["failures"]:,} / {r["shots"]:,}',
            f'{r["batch_us_per_shot"]:.2f}',f'{r["batch_us_per_shot"]/5:.2f}'),cols,accent=k=="frozen",size=10.9)
    gain=100*(1-hybrid["frozen"]["failures"]/hybrid["native"]["failures"])
    slowdown=hybrid["frozen"]["batch_us_per_shot"]/hybrid["native"]["batch_us_per_shot"]
    p.text(RIGHT, 674, f'{gain:.1f}% fewer errors at {slowdown:.1f}× the batch decoding time.',11,bold=True,color=TEAL)

    p.line(LEFT, 699, W-2*LEFT)
    p.text(LEFT, 723, "Reading the slosh from the calibration signal", 18, bold=True)
    p.text(LEFT, 742, f'Full FFT with phase matched {fft["fft"]["exact_grid"]}/49 fitted times; the strongest tone matched {fft["single_tone"]["exact_grid"]}/49. Noiseless simulation.', 10.6, color=MUTED)
    img_path = EXPERIMENTS / "bath-fft-timing-02/timing.png"
    img = ImageReader(str(img_path))
    iw, ih = img.getSize()
    w = min(658, 166 * iw/ih)
    c.drawImage(img, LEFT, H-916, width=w, height=w*ih/iw, mask="auto")
    example = csv_rows("bath-fft-timing-02/per_bath.csv")[0]
    p.text(LEFT, 930, f'For bath {example["bath"]}, the strongest tone is {float(example["tone_frequency"]):.4f} cycles/time unit. FFT and fit choose {float(example["fft_time"]):.2f}.', 10.8, color=TEAL)
    p.text(LEFT, 946, f'At 10,000 calibration shots per time and observable, {100*noisy_fft["within_one_step"]/noisy_fft["records"]:.1f}% of FFT choices fell within one 0.06-step of the fit.', 10.1, color=MUTED)
    x = 741
    p.text(x, 723, "Why dual rail helps", 17, bold=True)
    p.text(x, 748, "Northrop Grumman’s RQL-controlled dual-rail architecture", 11.4)
    p.text(x, 765, "encodes one excitation across two rails. An occupancy check", 11.4)
    p.text(x, 782, "can flag loss while preserving which-rail information. Both", 11.4)
    p.text(x, 799, "schedules use the same simulated hardware, calibration signals", 11.4)
    p.text(x, 816, "and check count. The bath map keeps the timing and phase.", 11.4)
    p.text(x, 845, "After the catch: common phase flips", 13, bold=True, color=TEAL)
    d={r["pulses"]:r for r in pulse["summary"]}
    p.text(x, 865, f'In an ideal finite hold, mean endpoint loss fell from {100*d[0]["final_loss_given_catch"]:.2f}%', 11.4)
    p.text(x, 882, f'to {100*d[128]["final_loss_given_catch"]:.3f}% with 128 pulses. This physical loss metric', 11.4)
    p.text(x, 899, "is conditional on the catch. Surface-code rates above use no pulse train.", 10.7)
    release = {r["second_interval_pulses"]:r["mean_peak_loss"] for r in pulse["release_after_128"]}
    p.text(x, 923, f'Peak loss in the next interval averaged {100*release[0]:.2f}% when pulses', 10.5,color=MUTED)
    p.text(x, 938, f'stopped and {100*release[128]:.3f}% when they continued.', 10.5,color=MUTED)
    p.line(LEFT, 955, W-2*LEFT)
    p.text(LEFT, 975, "¹ Check times are model-time units. Warmed CPU inference uses 4,000-shot batches in the bath noise sweep and 512-shot batches in the schedule controls. Accuracy uses the full counts shown.", 10.1, color=MUTED)
    p.text(LEFT, 993, "Fixed Stim results use provisional simple_uniform noise. Bath scheduling changes the physical channel. The decoder-only and bath studies answer different benchmark questions.", 10.2, color=MUTED)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
