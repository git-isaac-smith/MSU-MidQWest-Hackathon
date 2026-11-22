"""Synthetic examples for paired full-MAP reporting."""

import csv
import json

import pytest

from full_map_report import LABELS, summarize, write_report


def _example():
    truth = [1, 1, 0, 0]
    predictions = dict(mwpm=[1, 0, 0, 0], balanced=[0, 0, 1, 0], accuracy=[0, 1, 0, 0],
                       full_map=[0, 1, 0, 0])
    posterior = [0.1, 0.7, 0.5, 0.2]
    return truth, predictions, posterior, [99, 13, 13, 99]


def test_pooled_and_seed_observations_use_sampled_truth_and_model_risk_uses_posterior():
    summary = summarize(*_example(), timings=None)
    rows = {row["mode"]: row for row in summary["rows"]}
    assert {mode: row["errors"] for mode, row in rows.items()} == dict(mwpm=1, balanced=3, accuracy=1, full_map=1)
    assert rows["full_map"]["error_percent"] == 25
    assert rows["full_map"]["expected_error_percent"] == pytest.approx(27.5)
    assert rows["mwpm"]["expected_error_percent"] == pytest.approx(57.5)
    assert rows["mwpm"]["expected_excess_vs_full_map_percentage_points"] == pytest.approx(30)
    assert rows["balanced"]["expected_excess_vs_full_map_percentage_points"] == pytest.approx(10)
    assert rows["accuracy"]["expected_excess_vs_full_map_percentage_points"] == 0
    per_seed = {row["seed"]: row for row in rows["full_map"]["per_seed"]}
    assert per_seed[13]["errors"] == 0
    assert per_seed[99]["errors"] == 1
    assert per_seed[99]["total"] == 2


def test_paired_report_counts_gains_and_losses_and_bounds_zero_discordance():
    summary = summarize(*_example(), timings=None)
    pairs = {pair["reference"]: pair for pair in summary["paired_comparisons"]}
    mwpm, balanced, prior = (pairs[mode] for mode in ("mwpm", "balanced", "accuracy"))
    assert (mwpm["gains"], mwpm["losses"], mwpm["difference_percentage_points"]) == (1, 1, 0)
    assert mwpm["per_seed"] == [dict(seed=13, total=2, gains=1, losses=0), dict(seed=99, total=2, gains=0, losses=1)]
    assert (balanced["gains"], balanced["losses"], balanced["difference_percentage_points"]) == (2, 0, -50)
    assert "exact" in balanced["paired_ci_method"]
    assert (prior["gains"], prior["losses"], prior["mcnemar_two_sided_p"]) == (0, 0, 1)
    probability_upper = 1 - 0.05 ** 0.25
    assert prior["paired_ci95_percentage_points"] == pytest.approx([-100 * probability_upper, 100 * probability_upper])


def test_degenerate_many_discordant_pairs_still_have_an_interval():
    n = 40
    summary = summarize([0] * n, dict(mwpm=[1] * n, balanced=[0] * n, accuracy=[0] * n, full_map=[0] * n),
                        [0.1] * n, [27] * n, timings=None)
    pair = summary["paired_comparisons"][0]
    assert pair["difference_percentage_points"] == -100
    low, high = pair["paired_ci95_percentage_points"]
    assert low == -100 < high < 0


def test_full_map_uses_mwpm_within_numerical_tolerance_and_the_maximum_elsewhere():
    truth, predictions, posterior, seeds = _example()
    posterior[2] += 5e-13
    row = summarize(truth, predictions, posterior, seeds, None)["rows"][-1]
    assert row["expected_error_percent"] == pytest.approx(27.5)
    predictions["mwpm"][2] = 1
    predictions["full_map"][2] = 1
    rows = summarize(truth, predictions, posterior, seeds, None)["rows"]
    assert rows[2]["expected_excess_vs_full_map_percentage_points"] == 0
    predictions["mwpm"][2] = 0
    with pytest.raises(ValueError, match="numerical tie"):
        summarize(truth, predictions, posterior, seeds, None)
    predictions["full_map"][2] = 0
    predictions["full_map"][1] = 0
    with pytest.raises(ValueError, match="posterior maximum"):
        summarize(truth, predictions, posterior, seeds, None)


def test_report_writes_both_tables_and_the_observed_error_figure(tmp_path):
    timings = {mode: dict(shots=4, seconds=0.5, batch_size=2) for mode in LABELS}
    summary = summarize(*_example(), timings=timings)
    write_report(tmp_path, summary)
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == summary
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21
    map_row = next(row for row in rows if row["scope"] == "pooled" and row["mode"] == "full_map")
    assert float(map_row["shots_per_second"]) == 8
    assert float(map_row["error_percent"]) == 25
    assert float(map_row["expected_error_percent"]) == pytest.approx(27.5)
    assert (tmp_path / "full-map-errors.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
