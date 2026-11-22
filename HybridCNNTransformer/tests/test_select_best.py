"""select_best.py: per-distance hyperparameter selection from experiment results."""
import json
import os

import pytest
import yaml

import select_best as sb
from qecdec.config import from_dict


def row(model, name, params, finals, d_set=(3, 5), mwpm=0.01):
    d_set = list(d_set)
    return {"model": model, "group": name.split("/")[0], "name": name, "params": params, "status": "ok",
            "d_set": d_set, "final_by_d": {str(d): v for d, v in zip(d_set, finals)},
            "mwpm_by_d": {str(d): mwpm for d in d_set}}


def make_rows():
    h = "hybrid"
    return [
        row(h, "baseline/default", {"seed": 0}, [0.030, 0.030]),
        row(h, "seed/1", {"seed": 1}, [0.031, 0.031]),
        row(h, "seed/2", {"seed": 2}, [0.029, 0.029]),
        row(h, "lr/5e-4", {"lr": 5e-4}, [0.020, 0.040]),             # d=3 clearly better, d=5 clearly worse
        row(h, "lr/2e-3", {"lr": 2e-3}, [0.0295, 0.029]),            # within noise
        row(h, "width/96", {"width": 96}, [0.0295, 0.0295]),         # within noise everywhere
        row(h, "ablation/no_boundary_tokens", {"use_boundary_tokens": False}, [0.010, 0.010]),   # never combined
        row(h, "p/0.002", {"p": 0.002}, [0.001, 0.001]),             # different noise level: ignored
        row(h, "distance/d3", {"d_set": [3]}, [0.025], d_set=(3,)),
        row("cnn_baseline", "baseline/default", {"seed": 0}, [0.05, 0.05]),
        row("cnn_baseline", "lr/5e-4", {"lr": 5e-4}, [0.01, 0.01]),
    ]


class Args:
    metric, sigma, exclude_groups = "ratio", 2.0, sb.DEFAULT_EXCLUDE


def hybrid(rows):
    return [r for r in rows if r["model"] == "hybrid"]


def test_scores_noise_and_choice_per_distance():
    r3 = sb.analyze(hybrid(make_rows()), "hybrid", [3, 5], 3, Args)
    assert r3["baseline"]["n"] == 3 and r3["baseline"]["mean"] == pytest.approx(3.0)
    assert r3["baseline"]["noise"] == pytest.approx(0.1)
    verdict = {g["group"]: g["verdict"] for g in r3["groups"]}
    assert verdict["lr"] == "better" and verdict["width"] == "within noise"
    assert r3["overrides"] == {"lr": 5e-4}                       # ablation better but never combined
    assert r3["best_single"]["name"] == "ablation/no_boundary_tokens"
    assert "p" not in verdict and r3["distance_runs"][0]["name"] == "distance/d3"
    r5 = sb.analyze(hybrid(make_rows()), "hybrid", [3, 5], 5, Args)
    assert r5["overrides"] == {}                                 # lr is worse at d=5


def test_exclude_groups_flag_allows_combining_ablation():
    class A(Args):
        exclude_groups = []
    res = sb.analyze(hybrid(make_rows()), "hybrid", [3, 5], 3, A)
    assert res["overrides"] == {"lr": 5e-4, "use_boundary_tokens": False}


def test_no_baseline_and_zero_mwpm():
    rows = [row("hybrid", "lr/5e-4", {"lr": 5e-4}, [0.02, 0.02]),
            row("hybrid", "lr/2e-3", {"lr": 2e-3}, [0.03, 0.03], mwpm=0.0)]
    res = sb.analyze(rows, "hybrid", [3, 5], 3, Args)
    assert res["baseline"]["mean"] is None
    assert [g["verdict"] for g in res["groups"]] == ["no baseline"] and res["overrides"] == {}


def test_check_config_rejects_bad_width():
    ok, _ = sb.check_config(sb.build_config("hybrid", 3, {"width": 96}))
    assert ok
    ok, msg = sb.check_config(sb.build_config("hybrid", 3, {"width": 30}))
    assert not ok and "INVALID" in msg


def test_main_writes_outputs(tmp_path):
    src = tmp_path / "sweep"
    src.mkdir()
    (src / "results.json").write_text(json.dumps(
        {"meta": {"args": {"d_set": [3, 5]}}, "results": make_rows()}))
    out = tmp_path / "out"
    summary = sb.main([str(src), "--out-dir", str(out), "--d", "3"])
    run_dir = next(out.iterdir())
    for name in ("summary.txt", "best_hyperparameters.json", "per_group.csv"):
        assert (run_dir / name).is_file()
    cfg = yaml.safe_load((run_dir / "configs" / "hybrid_d3.yaml").read_text())
    assert cfg["lr"] == 5e-4 and cfg["d_set"] == [3] and cfg["out_dir"] == "runs/best_hybrid_d3"
    from_dict(cfg)
    assert (run_dir / "configs" / "cnn_d3.yaml").is_file()
    assert set(summary["sweep"]) == {"hybrid", "cnn_baseline"} and set(summary["sweep"]["hybrid"]) == {"3"}
