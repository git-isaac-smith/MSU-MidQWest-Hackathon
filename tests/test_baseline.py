import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from bascd.benchmark import run_case
from bascd.stats import shot_failures, wilson95


def test_scoring_counts_shots_and_rejects_broadcasting():
    truth = np.zeros((3, 2), dtype=bool)
    assert shot_failures(np.array([[1, 1], [0, 0], [0, 1]]), truth).tolist() == [True, False, True]
    with pytest.raises(ValueError):
        shot_failures(np.zeros((3, 1)), truth)


def test_wilson_bounds():
    assert wilson95(0, 10) == pytest.approx((0.0, 0.27753), abs=1e-5)
    assert wilson95(10, 10) == pytest.approx((0.72247, 1.0), abs=1e-5)
    assert wilson95(5, 10) == pytest.approx((0.23659, 0.76341), abs=1e-5)


@pytest.mark.parametrize("basis", ["x", "z"])
def test_noiseless_memories(basis):
    case = run_case(basis=basis, distance=3, rounds=3, p=0, shots=32, seed=5)
    assert case["paired"]["both_correct"] == 32
    for row in case["results"]:
        assert row["num_detectors"] == 24
        assert row["num_observables"] == 1
        assert row["failures"] == 0


def test_cli_writes_paired_results(tmp_path):
    run = subprocess.run(
        [sys.executable, "-m", "bascd", "--shots", "256", "--p", "0.01",
         "--output-dir", str(tmp_path)], capture_output=True, text=True, check=True,
    )
    report = json.loads(run.stdout)
    assert report == json.loads((tmp_path / "benchmark.json").read_text())
    with (tmp_path / "benchmark.csv").open(newline="") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 4
    for index, case in enumerate(report["cases"]):
        baseline, correlated = case["results"]
        paired = case["paired"]
        assert baseline["seed"] == correlated["seed"] == 101 + index
        assert baseline["basis"] == correlated["basis"] == ("x", "z")[index]
        assert sum(paired.values()) == 256
        assert baseline["failures"] == paired["mwpm_only_failed"] + paired["both_failed"]
        assert correlated["failures"] == paired["correlated_only_failed"] + paired["both_failed"]
        assert int(rows[2 * index]["failures"]) == baseline["failures"]
