import argparse
import csv
import json
from pathlib import Path

from .benchmark import run_case


def main():
    parser = argparse.ArgumentParser(description="Run paired Stim/PyMatching surface-code memories.")
    parser.add_argument("--basis", choices=("x", "z", "both"), default="both")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, help="defaults to the code distance")
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--shots", type=int, default=1000, help="shots per memory basis")
    parser.add_argument("--seed", type=int, default=101, help="first basis seed")
    parser.add_argument("--output-dir", type=Path, help="write benchmark.json and benchmark.csv")
    args = parser.parse_args()
    bases = ("x", "z") if args.basis == "both" else (args.basis,)
    rounds = args.distance if args.rounds is None else args.rounds
    try:
        cases = [
            run_case(basis=basis, distance=args.distance, rounds=rounds, p=args.p,
                     shots=args.shots, seed=args.seed + index)
            for index, basis in enumerate(bases)
        ]
    except ValueError as error:
        parser.error(str(error))

    report = json.dumps({"cases": cases}, indent=2)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "benchmark.json").write_text(report + "\n", encoding="utf-8")
        rows = [row for case in cases for row in case["results"]]
        with (args.output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(report)


if __name__ == "__main__":
    main()
