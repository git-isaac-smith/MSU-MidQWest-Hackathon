"""Draw a patch and first-round check trace from an archived Stim circuit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import stim

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INK = "#193044"
MUTED = "#526779"
GRID = "#dfe8ef"
DATA = "#526779"
X_CHECK = "#287d88"
Z_CHECK = "#67569c"
TRACE = "#ae482e"


def first_round(circuit: stim.Circuit):
    """Read the tick of each gate through the first ancilla measurement."""
    operations = []
    tick = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            break
        name = instruction.name
        if name == "TICK":
            tick += 1
        elif name in {"R", "H", "CX", "MR"}:
            targets = [target.value for target in instruction.targets_copy()]
            operations.append((tick, name, targets))
            if name == "MR":
                return operations, set(targets)
    raise ValueError("Expected an explicit first round ending in MR")


def draw_patch(ax, coordinates, checks, x_checks, ancilla, partners):
    for partner in partners:
        ax.plot(*zip(coordinates[ancilla], coordinates[partner]), color=TRACE,
                linewidth=2, alpha=0.65, zorder=2)
    for qubit, (x, y) in coordinates.items():
        if qubit in checks:
            color = TRACE if qubit == ancilla else X_CHECK if qubit in x_checks else Z_CHECK
            ax.scatter(x, y, marker="s", s=390, color=color, zorder=3)
            label_color = "white"
        else:
            ax.scatter(x, y, s=390, facecolor="white", edgecolor=DATA, linewidth=1.25, zorder=3)
            label_color = INK
        ax.text(x, y, str(qubit), color=label_color, ha="center", va="center",
                fontsize=10, weight="bold", zorder=4)

    xs, ys = zip(*coordinates.values())
    ax.set(xlim=(min(xs) - 0.55, max(xs) + 0.55), ylim=(max(ys) + 0.55, min(ys) - 0.55),
           xticks=sorted(set(xs)), yticks=sorted(set(ys)), aspect="equal")
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    ax.set_xlabel("Stim x coordinate", color=MUTED, labelpad=4, fontsize=10)
    ax.set_ylabel("Stim y coordinate", color=MUTED, labelpad=4, fontsize=10)
    ax.grid(color=GRID, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_trace(ax, operations, ancilla):
    last_tick = operations[-1][0]
    ax.set(xlim=(-0.88, last_tick + 0.55), ylim=(-1.03, 1.55))
    ax.axis("off")
    ax.hlines(0, -0.18, last_tick + 0.34, color=TRACE, linewidth=1.65)
    ax.text(-0.36, 0, f"q{ancilla}", color=TRACE, weight="bold", ha="right", va="center", fontsize=12)

    for tick in range(last_tick + 1):
        ax.vlines(tick, -0.52, 1.08, color=GRID, linewidth=0.8, zorder=0)
        ax.text(tick, -0.65, str(tick), color=MUTED, ha="center", va="center", fontsize=10)
    ax.text(last_tick / 2, -0.95, "Stim tick (zero-based)", color=MUTED, ha="center", fontsize=10)

    for tick, name, targets in operations:
        if ancilla not in targets:
            continue
        if name == "CX":
            for control, target in zip(targets[::2], targets[1::2], strict=True):
                if ancilla not in (control, target):
                    continue
                partner = target if ancilla == control else control
                ax.text(tick, 1.19, f"q{partner}", color=INK, weight="bold",
                        ha="center", va="center", fontsize=11)
                ax.vlines(tick, 0, 0.88, color=TRACE, linewidth=1.65)
                control_y, target_y = (0, 0.88) if ancilla == control else (0.88, 0)
                ax.scatter(tick, control_y, s=42, color=TRACE, zorder=3)
                ax.scatter(tick, target_y, s=135, facecolor="white", edgecolor=TRACE,
                           linewidth=1.5, zorder=3)
                ax.scatter(tick, target_y, s=67, marker="+", color=TRACE, linewidth=1.4, zorder=4)
        else:
            ax.text(tick, 0, name, color="white", weight="bold", fontsize=11,
                    ha="center", va="center", zorder=5,
                    bbox={"boxstyle": "round,pad=0.32", "facecolor": TRACE, "edgecolor": TRACE})


def plot_circuit(archive: Path, case_name: str, ancilla: int) -> None:
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    case = next((item for item in manifest["cases"] if item["case"] == case_name), None)
    if case is None:
        raise ValueError(f"Case {case_name!r} is missing from the archive manifest")
    circuit = stim.Circuit.from_file(archive / "cases" / case_name / "circuit.stim")
    operations, checks = first_round(circuit)
    if ancilla not in checks:
        raise ValueError(f"q{ancilla} is not measured in the first MR instruction")
    coordinates = {qubit: tuple(xy) for qubit, xy in circuit.get_final_qubit_coordinates().items()}
    x_checks = {qubit for _, op, qubits in operations if op == "H" for qubit in qubits} & checks
    basis = "X" if ancilla in x_checks else "Z"
    partners = [
        target if control == ancilla else control
        for _, op, qubits in operations if op == "CX"
        for control, target in zip(qubits[::2], qubits[1::2], strict=True)
        if ancilla in (control, target)
    ]
    round_count = sum(instruction.name == "MR" for instruction in circuit.flattened())

    fig = plt.figure(figsize=(12.3, 5.65), facecolor="white")
    fig.text(0.055, 0.946, "Rotated surface-code memory circuit", color=INK,
             va="top", fontsize=23, weight="bold")
    fig.text(0.055, 0.868,
             f"Distance {case['distance']}  /  {case['basis'].upper()} memory  /  "
             f"p = {case['p']:g}  /  {round_count} check rounds", color=MUTED, va="top", fontsize=13)
    fig.text(0.057, 0.765, "Physical patch", color=INK, fontsize=15, weight="bold")
    fig.text(0.435, 0.765, f"Round 1 gates on q{ancilla} ({basis} check)",
             color=INK, fontsize=15, weight="bold")

    draw_patch(fig.add_axes((0.060, 0.231, 0.285, 0.490)), coordinates, checks, x_checks, ancilla, partners)
    draw_trace(fig.add_axes((0.424, 0.265, 0.532, 0.420)), operations, ancilla)

    for x, label, color in [(0.057, "Data", DATA), (0.125, "X check", X_CHECK),
                            (0.214, "Z check", Z_CHECK), (0.302, f"q{ancilla} trace", TRACE)]:
        fig.text(x, 0.123, label, color=color, fontsize=11, weight="bold")
    gate_key = "R  reset     MR  measure and reset in Z"
    if basis == "X":
        gate_key += "     H  Hadamard"
    fig.text(0.435, 0.159, gate_key, color=MUTED, fontsize=11)
    fig.text(0.435, 0.103, "CX  filled dot = control, circled plus = target", color=MUTED, fontsize=11)
    fig.text(0.057, 0.042, "Gate trace omits noise. The full Stim timeline includes all rounds and noise.",
             color=MUTED, fontsize=10)

    overview = archive / "circuit-overview.png"
    timeline = archive / "circuit-timeline.svg"
    fig.savefig(overview, dpi=170)
    plt.close(fig)
    timeline.write_text(str(circuit.diagram("timeline-svg")), encoding="utf-8")
    print(f"Wrote {overview} and {timeline}")
    for other_case in manifest["cases"]:
        if (other_case["distance"], other_case["p"]) != (case["distance"], case["p"]):
            continue
        if other_case["basis"] == case["basis"]:
            continue
        other_circuit = stim.Circuit.from_file(archive / "cases" / other_case["case"] / "circuit.stim")
        other_timeline = archive / f"circuit-timeline-{other_case['basis'].lower()}.svg"
        other_timeline.write_text(str(other_circuit.diagram("timeline-svg")), encoding="utf-8")
        print(f"Wrote {other_timeline}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--case", default="d3-z-p0.005", help="Case name from the archive manifest")
    parser.add_argument("--ancilla", type=int, default=18, help="Check ancilla to trace")
    args = parser.parse_args()
    plot_circuit(args.archive, args.case, args.ancilla)


if __name__ == "__main__":
    main()
