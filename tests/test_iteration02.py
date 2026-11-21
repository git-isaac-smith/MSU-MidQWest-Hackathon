import csv
import inspect
import json

import numpy as np
import pymatching
import pytest
from jsonschema import ValidationError

from bascd import search
from bascd.decoders import CHOICES, CompiledDecoder, DecoderSpec, Policy, compile_decoder, validator
from bascd.experiment import main
from bascd.measure import SPLIT_TAGS, Scenario, measure, sample
from bascd.records import scenarios
from bascd.stats import paired_counts


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_policy_roundtrip_and_distance_dispatch():
    policy = Policy((CHOICES[0], CHOICES[1], CHOICES[2]))
    assert Policy.from_dict(policy.to_dict()) == policy
    assert [policy.at(d) for d in (3, 5, 7)] == list(policy.specs)
    assert Policy.from_dict(policy.to_dict()).policy_id == policy.policy_id


@pytest.mark.parametrize("entry", [
    {"decoder_type": "bp"}, {"decoder_type": "mwpm", "boundary_scale": 1},
    {"decoder_type": "weighted_mwpm", "temporal_scale": 1, "boundary_scale": 1},
    {"decoder_type": "weighted_mwpm", "temporal_scale": 1.25, "boundary_scale": False},
    {"decoder_type": "correlated_mwpm", "temporal_scale": 1.25},
    {"decoder_type": "weighted_mwpm", "temporal_scale": 0, "boundary_scale": 1},
])
def test_proposal_rejects_unsupported_controls(entry):
    value = Policy((CHOICES[0],) * 3).to_dict()
    value["by_distance"]["5"] = entry
    with pytest.raises(ValidationError):
        Policy.from_dict(value)


def test_typed_config_validation_and_strict_json():
    with pytest.raises(ValueError):
        DecoderSpec("weighted_mwpm", True, 1.25)
    with pytest.raises(ValueError):
        DecoderSpec("correlated_mwpm", 1, 1.25)
    with pytest.raises(ValueError):
        Policy(())
    for raw in ('{"a":1,"a":2}', '{"a": NaN}', '{"a":1e999}'):
        with pytest.raises(ValueError):
            search.parse_proposal(raw)


def test_graph_adjustment_preserves_topology_and_changes_inference():
    case = Scenario(3, 0.01, "x")
    data = sample(case, 42, "train", 4096)
    base = compile_decoder(data.model, CHOICES[0])
    weighted = compile_decoder(data.model, DecoderSpec("weighted_mwpm", 0.75, 1.25))
    coords = data.model.get_detector_coordinates()
    edges = {(a, b): attr for a, b, attr in weighted.graph.edges()}
    assert len(edges) == base.graph.num_edges
    groups = set()
    for a, b, raw in base.graph.edges():
        scale = 1.25 if b is None else 0.75 if coords[a][:2] == coords[b][:2] else 1
        groups.add(scale)
        assert edges[a, b]["weight"] == pytest.approx(scale * raw["weight"])
        assert edges[a, b]["fault_ids"] == raw["fault_ids"]
    assert groups == {0.75, 1, 1.25}
    assert np.any(base.decode_batch(data.detectors) != weighted.decode_batch(data.detectors))
    assert sample(case, 42, "train", 4096).metadata["sample_sha256"] == data.metadata["sample_sha256"]


def test_inference_contract_and_reference_predictions():
    data = sample(Scenario(3, 0.01, "z"), 17, "sweep", 256)
    assert list(inspect.signature(CompiledDecoder.decode_batch).parameters) == ["self", "detectors"]
    assert not data.truth.flags.writeable and not data.detectors.flags.writeable
    for spec in CHOICES[:2]:
        correlated = spec == CHOICES[1]
        native = pymatching.Matching.from_detector_error_model(data.model, enable_correlations=correlated)
        pred = compile_decoder(data.model, spec).decode_batch(data.detectors)
        assert pred.shape == (256, 1)
        np.testing.assert_array_equal(pred, native.decode_batch(data.detectors, enable_correlations=correlated))
        row, failures = measure(data, spec, 2, 16)
        np.testing.assert_array_equal(failures, np.any(pred != data.truth, axis=1))
        assert len(row["batch_times_s"]) == 2
        assert 0 < row["single_p50_us"] <= row["single_p95_us"]


def test_seeds_are_separated_and_final_split_is_inaccessible():
    seeds = [case.seed(2026, split) for case in scenarios() for split in SPLIT_TAGS]
    assert len(set(seeds)) == 96
    case = Scenario(3, 0.01, "x")
    with pytest.raises(ValueError, match="reserved"):
        sample(case, 2026, "final_test", 10)
    assert sample(case, 2026, "train", 256).metadata["sample_sha256"] != sample(
        case, 2026, "validation", 256).metadata["sample_sha256"]


def test_statistics_and_empirical_pareto():
    a, b = np.array([1, 1, 0, 0], bool), np.array([0, 1, 0, 1], bool)
    assert paired_counts(a, b) == dict(mwpm_only_failed=1, correlated_only_failed=1,
                                        both_failed=1, both_correct=1)
    rows = [{"shots": 100, "failures": k, "median_batch_s": 0.001} for k in (0, 20)]
    summary = search.summarize(rows)
    assert summary["ler"] == pytest.approx(0.1)
    assert summary["batch_us_per_shot"] == pytest.approx(10)
    assert summary["ler95_low"] < 0.1 < summary["ler95_high"]
    points = [(2, 2), (1, 3), (1, 3), (2, 4), (3, 2)]
    trials = [dict(trial=i, status="ok", validation=dict(ler=x/100, batch_us_per_shot=y))
              for i, (x, y) in enumerate(points)]
    assert search.pareto([*trials, {"status": "failed"}]) == [0, 1, 2]


def test_random_search_replay_and_unique_budget():
    first = search.random_policies(17, 9)
    assert first == search.random_policies(17, 9)
    assert first != search.random_policies(18, 9)
    assert len({p.policy_id for p in first}) == 9


def test_guided_campaign_budget_logging_and_blind_validation(tmp_path, monkeypatch):
    out = tmp_path / "guided"
    seen_splits = []
    original = search.sample

    def observe(case, seed, split, shots):
        seen_splits.append(split)
        return original(case, seed, split, shots)

    monkeypatch.setattr(search, "sample", observe)
    main(["search", "--out", str(out), "--strategy", "guided", "--budget", "3", "--shots", "40", "--repeats", "1"])
    proposal = tmp_path / "policy.json"
    policy = Policy((CHOICES[0], CHOICES[1], CHOICES[2]))
    proposal.write_text(json.dumps(policy.to_dict()))
    main(["validate", str(proposal)])
    for _ in range(2):
        main(["submit", "--out", str(out), str(proposal)])
    assert set(seen_splits) == {"train"}
    assert not (out / "validation-references.json").exists()
    proposal.write_text("{bad-json")
    main(["submit", "--out", str(out), str(proposal)])
    assert set(seen_splits) == {"train", "validation"}
    history = read(out / "history.json")
    assert [t["status"] for t in history] == ["ok", "invalid", "invalid"]
    assert history[1]["error"] == "duplicate policy"
    assert history[2]["proposal"] == "{bad-json"
    assert history[0]["validation"]["shots"] == 24 * 40
    validator("feedback").validate(read(out / "feedback.json"))
    assert "validation" not in read(out / "feedback.json")["trials"][0]
    with (out / "history.csv").open(newline="") as f:
        assert [t["status"] for t in csv.DictReader(f)] == ["ok", "invalid", "invalid"]
    refs = read(out / "validation-references.json")
    for row in read(out / "measurements/001-validation.json"):
        for ref in (r for r in refs if r["sample_sha256"] == row["sample_sha256"]):
            kind = ref["decoder"]
            assert ref["failures"] - row["failures"] == (
                row[f"{kind}_only_failed"] - row[f"candidate_only_failed_vs_{kind}"])
    with pytest.raises(SystemExit):
        main(["submit", "--out", str(out), str(proposal)])
    before = (out / "history.json").read_bytes()
    main(["resume", "--out", str(out)])
    assert (out / "history.json").read_bytes() == before


def test_evaluation_failures_consume_budget(tmp_path, monkeypatch):
    out = tmp_path / "random"
    def fail(*args):
        raise RuntimeError("decoder failed")
    monkeypatch.setattr(search, "evaluate", fail)
    main(["search", "--out", str(out), "--budget", "1", "--shots", "20", "--repeats", "1"])
    [trial] = read(out / "history.json")
    assert trial["status"] == "failed" and "decoder failed" in trial["error"]
    assert read(out / "pareto.json") == []


def test_sweep_cli_replay_and_no_overwrite(tmp_path):
    args = ["--distances", "3", "--probabilities", "0.01", "--shots", "128",
            "--latency-shots", "8", "--repeats", "1"]
    reports = []
    for run in ("first", "second"):
        out = tmp_path / run
        main(["sweep", "--out", str(out), *args])
        reports.append(read(out / "benchmark.json"))
        assert all((out / name).is_file() for name in
                   ("benchmark.csv", "paired.csv", "logical-error-rate.png", "inference-runtime.png"))
    for left, right in zip(reports[0]["measurements"], reports[1]["measurements"], strict=True):
        assert left["sample_sha256"] == right["sample_sha256"] and left["failures"] == right["failures"]
    for case in reports[0]["paired"]:
        assert sum(case[k] for k in ("both_correct", "both_failed", "mwpm_only_failed", "correlated_only_failed")) == 128
        assert case["disagreements"] == case["mwpm_only_failed"] + case["correlated_only_failed"]
    with pytest.raises(SystemExit):
        main(["sweep", "--out", str(out), *args])
