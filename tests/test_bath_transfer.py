import csv
import json

import pytest

from bascd.bath import BathModel, BathParameters
from bascd.bath_initialize import channel_record
from bascd.bath_transfer import run
from bascd.decoders import CHOICES, compile_decoder
from bascd.measure import Scenario, sample
from bascd.stats import paired_ler_difference, shot_failures


def test_transfer_matches_an_independent_native_replay_and_preserves_pairs(tmp_path):
    model = BathModel(BathParameters(.3, .6, .2, .7, .001, .001), "split")
    clock, selected = model.initialize(4)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"format_version": 1, "clock": channel_record(clock),
                                   "selected": channel_record(selected)}), encoding="utf-8")
    case = Scenario(3, .01, "z")
    reference = sample(case, 67, "sweep", 256)
    out = tmp_path / "run"
    summary = run(profile, out, [case], 67, 256, 1, 4, ["test"])
    with (out / "benchmark.csv").open(newline="") as src:
        rows = list(csv.DictReader(src))
    assert len(rows) == len(summary) == 8
    assert {r["sample_sha256"] for r in rows} == {reference.metadata["sample_sha256"]}
    assert {r["circuit_sha256"] for r in rows} == {reference.metadata["circuit_sha256"]}
    assert len({r["hypothesis_dem_sha256"] for r in rows if r["prior"] != "expanded_uniform"}) == 3
    for spec in CHOICES[:2]:
        group = [r for r in summary if r["decoder"] == spec.decoder_type]
        native = next(r for r in group if r["prior"] == "native")
        control = next(r for r in group if r["prior"] == "expanded_uniform")
        predictions = compile_decoder(reference.model, spec).decode_batch(reference.detectors)
        assert native["failures"] == shot_failures(predictions, reference.truth).sum()
        for row in group:
            if row["prior"] != "native":
                pairs = row["paired_vs_native"]
                assert row["failures"] - native["failures"] == pairs["lost"] - pairs["recovered"]
            if row["prior"] in ("clock", "selected"):
                pairs = row["paired_vs_expanded_uniform"]
                assert row["failures"] - control["failures"] == pairs["lost"] - pairs["recovered"]
    with pytest.raises(FileExistsError):
        run(profile, out, [case], 67, 256, 1, 0, ["test"])


def test_paired_heterogeneous_strata_use_shot_weights_and_detect_invalid_counts():
    result = paired_ler_difference([40, 160], [25, 2], [3, 15])
    assert result["ler_difference"] == pytest.approx(-9 / 200)
    low, high = result["approximate_95_ci"]
    assert low < result["ler_difference"] < high
    with pytest.raises(ValueError):
        paired_ler_difference([40], [25], [18])
