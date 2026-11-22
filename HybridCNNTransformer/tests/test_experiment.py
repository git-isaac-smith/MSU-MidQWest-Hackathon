"""experiment.py: the experiment list is well formed and every config builds and runs a forward pass."""
import argparse
import json
import os

import pytest
import torch

import experiment as ex
from qecdec.engine import TrainData
from qecdec.models import build_model

ARGS = argparse.Namespace(steps=40, d_set=[3, 5], val_shots=64, set=[], device="cpu")


def test_experiment_list_shape():
    names = [n for n, _ in ex.EXPERIMENTS]
    assert len(names) == len(set(names))
    for name, params in ex.EXPERIMENTS:
        assert "/" in name and isinstance(params, dict) and len(params) == 1, name


@pytest.mark.parametrize("model", list(ex.CONFIGS))
def test_every_experiment_builds_and_runs(model):
    for name, params in ex.EXPERIMENTS:
        if not ex.applies(model, params):
            continue
        cfg = ex.make_config(model, params, ARGS, "unused")
        cfg.batch_size, cfg.bf16 = 4, False
        net = build_model(cfg, max(cfg.d_set))
        batch = TrainData(cfg, torch.device("cpu")).next_batch(3)
        out = net(batch)
        assert out.shape == (4,) and torch.isfinite(out).all(), name


def test_model_specific_keys_are_skipped():
    assert not ex.applies("cnn_baseline", {"attn_layers": 2})
    assert not ex.applies("hybrid", {"dilations": [1, 2, 1]})
    assert ex.applies("hybrid", {"lr": 1e-3}) and ex.applies("cnn_baseline", {"lr": 1e-3})


def test_short_run_scaling_and_override():
    cfg = ex.make_config("hybrid", {"lr": 3e-4}, ARGS, "x")
    assert (cfg.steps, cfg.lr, cfg.d_set, cfg.warmup_steps, cfg.val_every) == (40, 3e-4, [3, 5], 4, 5)
    assert ex.make_config("hybrid", {"steps": 80}, ARGS, "x").val_every == 10
    assert ex.make_config("hybrid", {"d_set": [7]}, ARGS, "x").d_set == [7]


def test_list_and_group_filter(capsys):
    ex.main(["--list", "--model", "cnn", "--groups", "dilations", "baseline"])
    out = capsys.readouterr().out
    assert "dilations/1-2-1" in out and "baseline/default" in out and "attn_layers/" not in out
    assert "lr/" not in out and "hybrid " not in out
    with pytest.raises(SystemExit):
        ex.main(["--list", "--groups", "nope"])


@pytest.mark.smoke
def test_tiny_sweep_and_resume(tmp_path):
    flags = ["--model", "hybrid", "--groups", "baseline", "seed", "--steps", "24", "--d-set", "3",
             "--val-shots", "256", "--device", "cpu", "--out-dir", str(tmp_path), "--no-plot"]
    rows = ex.main(flags)
    assert [r["name"] for r in rows] == ["baseline/default", "seed/1", "seed/2"]
    assert all(r["status"] == "ok" and r["final_mean"] <= 0.5 and r["mwpm_mean"] >= 0 for r in rows)
    run_dir = os.path.dirname(rows[0]["out_dir"])
    assert os.path.isfile(os.path.join(run_dir, "results.csv"))
    # resume: nothing left to do, nothing retrained, nothing lost
    rows2 = ex.main(flags + ["--run-dir", run_dir])
    assert len(rows2) == 3
    with open(os.path.join(run_dir, "results.json")) as f:
        assert len(json.load(f)["results"]) == 3
