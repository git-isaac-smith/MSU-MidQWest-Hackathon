"""Plot the bath background sweep and fixed-circuit fault removals from measured counts."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
bath = json.loads((HERE.parent / "bath-background-stress-01/result.json").read_text())
with (HERE / "fault-sources/removals.csv").open(newline="") as file:
    removals = [row for row in csv.DictReader(file) if float(row["p"]) == .01]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.7))
fig.subplots_adjust(left=.085, right=.97, bottom=.23, top=.76, wspace=.57)
fig.text(.055, .94, "Which errors become visible under stress?", fontsize=18, fontweight="bold")
fig.text(.055, .875, "Distance-7 Z memory  ·  7 rounds  ·  20,000 shots per condition", color="#425367")

for arm, color, marker, label in zip(bath["arms"], ("#435977", "#b75420"), ("o", "D"),
                                      ("Fixed clock", "Bath-timed check"), strict=True):
    points = sorted((row for row in bath["points"] if row["arm"] == arm["arm"]), key=lambda row: row["p_check"])
    x = [100 * row["p_check"] for row in points]
    y = np.array([100 * row["logical_error_rate"] for row in points])
    lo, hi = 100 * np.array([row["wilson_95"] for row in points]).T
    left.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, capsize=3, label=label)
left.set(xscale="log", yscale="log", xlabel="Background check-error parameter (%)",
         ylabel="Logical failures (%)", ylim=(.005, 100))
left.set_xticks([.1, .5, 1, 2, 4], ["0.1", "0.5", "1", "2", "4"])
left.set_yticks([.01, .1, 1, 10, 50], ["0.01", "0.1", "1", "10", "50"])
left.set_title("Sakidja bath simulation", loc="left", fontweight="bold")
left.legend(frameon=False, loc="lower right")
left.grid(axis="y", color="#dce1e6", linewidth=.7)

names = ["All faults", "Remove gate faults", "Remove idle faults", "Remove readout flips", "Remove reset flips"]
raw = {r["removed"]: r for r in removals}
order = ("gate", "round_start_data", "readout", "reset")
counts = [int(raw["gate"]["full_failures"]), *(int(raw[k]["remaining_failures"]) for k in order)]
rates = 100 * np.array(counts) / int(raw["gate"]["shots"])
right.barh(names, rates, color=["#354963", *["#348477"] * 4], height=.56)
right.invert_yaxis()
right.set_xlim(0, max(rates) * 1.25)
right.set_xlabel("Logical failures (%)")
right.set_title("Fixed Stim circuit at 1% noise", loc="left", fontweight="bold")
for i, rate in enumerate(rates):
    right.text(rate + .12, i, f"{rate:.3f}%", va="center")
right.spines["left"].set_visible(False)
right.tick_params(axis="y", length=0)
fig.text(.055, .055, "Bath error bars show Wilson 95% intervals. Each fault removal uses the same underlying shots and decoder.",
         fontsize=9, color="#425367")
fig.savefig(HERE / "noise-stress.png", dpi=165, facecolor="white")
plt.close(fig)
