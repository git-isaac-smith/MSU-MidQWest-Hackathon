#!/usr/bin/env python
"""Hyperparameter experiments on short ("large smoke") training runs, for the CNN baseline and/or the hybrid.

Every experiment changes ONE hyperparameter relative to the base config and trains for --steps steps
(default 2000, a tenth of a full run). The base is configs/<model>.yaml with d_set=--d-set (default 3 5),
steps=--steps, val_shots=--val-shots, and warm-up / validation interval scaled to the shorter run.

    python experiment.py --list                                   # plan + rough run-time estimates, no training
    python experiment.py --model hybrid --groups lr ablation      # only some groups
    python experiment.py                                          # everything, both models

Experiments are the list EXPERIMENTS below: (name, {hyperparameter: value}); the group is the part of the
name before "/". Hybrid-only keys (attention, tokens, ...) are skipped for the CNN and vice versa.

Each invocation writes to its own folder  <out-dir>/<YYYYmmdd_HHMMSS>/  (never reused):
    <model>__<group>-<label>/   one training run each (config.json, log.jsonl, best.pt, training_curves.png)
    results.csv, results.json   one row per run, rewritten after every run
Every run is also scored against MWPM on the same validation shots, so runs with different p or d can be
compared through  final/mwpm  (1.0 = as good as PyMatching, lower is better). Validation sets are small, so
differences below the spread of the seed/* runs are noise. To continue an interrupted sweep (finished runs
are skipped):  python experiment.py --run-dir experiments/<timestamp> <same flags>
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import traceback

import yaml

import train as train_mod
from qecdec.config import config_to_dict, from_dict, load_config
from qecdec.data import ShotSource, build_mwpm, mwpm_predict
from qecdec.plotting import read_log
from qecdec.utils import new_run_dir

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {"hybrid": "configs/hybrid.yaml", "cnn_baseline": "configs/cnn_baseline.yaml"}
SHORT = {"hybrid": "hybrid", "cnn_baseline": "cnn"}

# ---------------------------------------------------------------------------------------------------------
# The experiments: (name, {hyperparameter: value}). Group = name.split("/")[0]. Baseline values: p 0.005,
# lr 1e-3, batch_size 256, weight_decay 0.01, width 64, attn_layers 3, heads 4, stem_layers 2, mlp_ratio 2,
# boundary_clip 3, dilations [1, 2, 4, 1], base d_set = --d-set.
# ---------------------------------------------------------------------------------------------------------
EXPERIMENTS = [
    # reference runs and run-to-run noise (same config, different seed)
    ("baseline/default", {"seed": 0}),
    ("seed/1", {"seed": 1}),
    ("seed/2", {"seed": 2}),
    # code distance(s) trained on (compare through final/mwpm; d=9 and d=11 are slow, see --list)
    ("distance/d3", {"d_set": [3]}),
    ("distance/d5", {"d_set": [5]}),
    ("distance/d7", {"d_set": [7]}),
    ("distance/d9", {"d_set": [9]}),
    ("distance/d11", {"d_set": [11]}),
    ("distance/d3-5-7", {"d_set": [3, 5, 7]}),
    # physical error rate used for training
    ("p/0.002", {"p": 0.002}),
    ("p/0.003", {"p": 0.003}),
    ("p/0.007", {"p": 0.007}),
    ("p/0.01", {"p": 0.01}),
    # learning rate
    ("lr/3e-4", {"lr": 3e-4}),
    ("lr/5e-4", {"lr": 5e-4}),
    ("lr/2e-3", {"lr": 2e-3}),
    ("lr/4e-3", {"lr": 4e-3}),
    # training shots per step (total shots seen = steps * batch_size)
    ("shots/b64", {"batch_size": 64}),
    ("shots/b128", {"batch_size": 128}),
    ("shots/b512", {"batch_size": 512}),
    ("shots/b1024", {"batch_size": 1024}),
    # training length
    ("steps/1000", {"steps": 1000}),
    ("steps/4000", {"steps": 4000}),
    # regularisation / optimisation
    ("weight_decay/0", {"weight_decay": 0.0}),
    ("weight_decay/0.1", {"weight_decay": 0.1}),
    ("grad_clip/0.5", {"grad_clip": 0.5}),
    ("grad_clip/5", {"grad_clip": 5.0}),
    # capacity
    ("width/32", {"width": 32}),
    ("width/96", {"width": 96}),
    ("width/128", {"width": 128}),
    # boundary distance clip (stabiliser-spacing units)
    ("boundary_clip/2", {"boundary_clip": 2}),
    ("boundary_clip/5", {"boundary_clip": 5}),
    # ablations of the boundary fix / mask-aware convolution
    ("ablation/no_boundary_channels", {"use_boundary_channels": False}),
    ("ablation/no_boundary_tokens", {"use_boundary_tokens": False}),      # hybrid only
    ("ablation/no_mask_aware_conv", {"mask_aware_conv": False}),
    # hybrid only: attention stage and stem
    ("attn_layers/1", {"attn_layers": 1}),
    ("attn_layers/2", {"attn_layers": 2}),
    ("attn_layers/4", {"attn_layers": 4}),
    ("heads/2", {"heads": 2}),
    ("heads/8", {"heads": 8}),
    ("stem_layers/1", {"stem_layers": 1}),
    ("stem_layers/3", {"stem_layers": 3}),
    ("mlp_ratio/4", {"mlp_ratio": 4}),
    # CNN only: dilation schedule (receptive field 1 + 2 * (1 + sum) must be >= the largest d)
    ("dilations/1-1-1-1", {"dilations": [1, 1, 1, 1]}),
    ("dilations/1-2-1", {"dilations": [1, 2, 1]}),
    ("dilations/1-2-4-2-1", {"dilations": [1, 2, 4, 2, 1]}),
]
GROUPS = list(dict.fromkeys(name.split("/")[0] for name, _ in EXPERIMENTS))
HYBRID_ONLY = {"use_boundary_tokens", "stem_layers", "attn_layers", "heads", "mlp_ratio", "n_fourier",
               "rel_clip", "bias_hidden"}
CNN_ONLY = {"dilations"}

# Rough seconds per training step at batch 256 (bf16, d uniform), measured on one laptop CPU (AMD 760M, no
# CUDA) for d = 3, 5, 7; larger d is extrapolated with a power law. Only used for the --list estimate.
_STEP_S = {"hybrid": {3: 0.107, 5: 0.282, 7: 0.648}, "cnn_baseline": {3: 0.189, 5: 0.594, 7: 1.301}}


def applies(model: str, params: dict) -> bool:
    keys = set(params)
    return not (keys & (HYBRID_ONLY if model == "cnn_baseline" else CNN_ONLY))


def step_seconds(model: str, d: int) -> float:
    t = _STEP_S[model]
    if d in t:
        return t[d]
    k = math.log(t[7] / t[5]) / math.log(7 / 5)
    return t[7] * (d / 7) ** k


def estimate_minutes(cfg) -> float:
    per_step = sum(step_seconds(cfg.model, d) for d in cfg.d_set) / len(cfg.d_set) * cfg.batch_size / 256
    return cfg.steps * per_step / 60


def make_config(model: str, params: dict, args, out_dir: str):
    """Base config for `model`, shortened to a smoke-size run, then the experiment's hyperparameter."""
    raw = config_to_dict(load_config(os.path.join(ROOT, CONFIGS[model])))
    raw.update(steps=args.steps, d_set=list(args.d_set), val_shots=args.val_shots, device=args.device)
    extra = {}
    for item in args.set:
        k, v = item.split("=", 1)
        extra[k.strip()] = yaml.safe_load(v)
    raw.update(extra)
    raw.update(copy.deepcopy(params))
    pinned = set(extra) | set(params)
    if "warmup_steps" not in pinned:
        raw["warmup_steps"] = max(1, raw["steps"] // 10)
    if "val_every" not in pinned:
        raw["val_every"] = max(1, raw["steps"] // 8)
    raw["log_every"] = max(1, min(raw["log_every"], raw["steps"] // 10))
    raw["out_dir"] = out_dir
    return from_dict(raw)


_mwpm_cache: dict = {}


def mwpm_val_ler(d: int, p: float, shots: int) -> float:
    """PyMatching's LER on exactly the validation shots train.py uses (seed 777_000_000 + d)."""
    key = (d, p, shots)
    if key not in _mwpm_cache:
        src = ShotSource(d, p, seed=777_000_000 + d)
        det, obs = src.sample(shots)
        _mwpm_cache[key] = float((mwpm_predict(build_mwpm(src.circuit), det) != obs).mean())
    return _mwpm_cache[key]


def run_one(model: str, name: str, params: dict, args, run_dir: str) -> dict:
    out_dir = os.path.join(run_dir, f"{SHORT[model]}__{name.replace('/', '-')}")
    cfg = make_config(model, params, args, out_dir)
    row = {"model": model, "group": name.split("/")[0], "name": name, "params": params, "status": "ok",
           "steps": cfg.steps, "batch_size": cfg.batch_size, "train_shots": cfg.steps * cfg.batch_size,
           "d_set": cfg.d_set, "p": cfg.p, "lr": cfg.lr, "seed": cfg.seed, "out_dir": out_dir}
    t0 = time.perf_counter()
    try:
        train_mod.run(cfg, plot=not args.no_plot)
        _, val = read_log(os.path.join(out_dir, "log.jsonl"))
        final, best = val[-1], min(val, key=lambda r: r["mean"])
        mwpm = {str(d): mwpm_val_ler(d, cfg.p, cfg.val_shots) for d in cfg.d_set}
        mwpm_mean = sum(mwpm.values()) / len(mwpm)
        row.update(final_mean=final["mean"], best_mean=best["mean"], best_step=best["step"],
                   final_by_d={str(d): final[str(d)] for d in cfg.d_set}, mwpm_mean=mwpm_mean, mwpm_by_d=mwpm,
                   ratio_final=final["mean"] / mwpm_mean if mwpm_mean > 0 else None)
    except Exception as e:                      # one bad config (e.g. out of memory) must not end the sweep
        traceback.print_exc()
        row.update(status="failed", error=repr(e))
    row["minutes"] = (time.perf_counter() - t0) / 60
    return row


def save_rows(run_dir: str, args, rows: list) -> None:
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"meta": {"timestamp": os.path.basename(run_dir), "args": vars(args), "argv": sys.argv},
                   "results": rows}, f, indent=2)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def print_summary(rows: list) -> None:
    print(f"\n{'experiment':<34}{'model':<13}{'final':>8}{'best':>8}{'@step':>7}{'mwpm':>8}{'final/mwpm':>11}{'min':>7}")
    print("-" * 96)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['name']:<34}{r['model']:<13}  FAILED: {r.get('error', '')[:50]}")
            continue
        ratio = "n/a" if r["ratio_final"] is None else f"{r['ratio_final']:.2f}"
        print(f"{r['name']:<34}{r['model']:<13}{r['final_mean']:>8.4f}{r['best_mean']:>8.4f}{r['best_step']:>7d}"
              f"{r['mwpm_mean']:>8.4f}{ratio:>11}{r['minutes']:>7.1f}")


def print_plan(plan: list, args) -> None:
    total = 0.0
    print(f"{'#':>3}  {'model':<13}{'experiment':<34}{'change':<28}{'steps':>6}{'est min':>9}")
    for i, (model, name, params) in enumerate(plan, 1):
        cfg = make_config(model, params, args, "")
        est = estimate_minutes(cfg)
        total += est
        print(f"{i:>3}  {model:<13}{name:<34}{json.dumps(params):<28}{cfg.steps:>6}{est:>9.1f}")
    print(f"\n{len(plan)} runs, about {total / 60:.1f} h in total (rough CPU estimate, validation excluded). "
          f"Groups: {', '.join(GROUPS)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["hybrid", "cnn", "both"], default="both")
    ap.add_argument("--groups", nargs="*", default=None, metavar="GROUP",
                    help=f"only these groups (default: all). Available: {', '.join(GROUPS)}")
    ap.add_argument("--list", action="store_true", help="print the selected runs with time estimates and exit")
    ap.add_argument("--steps", type=int, default=2000, help="training steps per run (full run: 20000)")
    ap.add_argument("--d-set", type=int, nargs="*", default=[3, 5], help="base training distances")
    ap.add_argument("--val-shots", type=int, default=4096, help="validation shots per distance")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra override applied to every run, e.g. p=0.004 (an experiment's own value wins)")
    ap.add_argument("--out-dir", default="experiments", help="root folder; each invocation gets a timestamped subfolder")
    ap.add_argument("--run-dir", default=None, help="continue in this existing experiment folder, skipping finished runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-plot", action="store_true", help="skip the per-run training_curves.png")
    args = ap.parse_args(argv)

    unknown = sorted(set(args.groups or []) - set(GROUPS))
    if unknown:
        ap.error(f"unknown group(s) {unknown}; available: {GROUPS}")
    models = list(CONFIGS) if args.model == "both" else ["hybrid" if args.model == "hybrid" else "cnn_baseline"]
    groups = set(args.groups or GROUPS)
    plan = [(m, name, params) for name, params in EXPERIMENTS if name.split("/")[0] in groups
            for m in models if applies(m, params)]
    if args.list:
        return print_plan(plan, args)

    rows = []
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isdir(run_dir):
            ap.error(f"--run-dir {run_dir} does not exist")
        path = os.path.join(run_dir, "results.json")
        if os.path.isfile(path):
            with open(path) as f:
                rows = [r for r in json.load(f)["results"] if r["status"] == "ok"]   # failed runs are retried
    else:
        run_dir = new_run_dir(args.out_dir)
    done = {(r["model"], r["name"]) for r in rows}
    todo = [e for e in plan if (e[0], e[1]) not in done]
    print(f"[exp] {len(todo)} runs to do ({len(done)} already finished) -> {run_dir}", flush=True)

    try:
        for i, (model, name, params) in enumerate(todo, 1):
            print(f"\n[exp] ===== run {i}/{len(todo)}: {model} {name} {json.dumps(params)}", flush=True)
            row = run_one(model, name, params, args, run_dir)
            rows.append(row)
            save_rows(run_dir, args, rows)
            if row["status"] == "ok":
                print(f"[exp] done in {row['minutes']:.1f} min: final {row['final_mean']:.4f} best "
                      f"{row['best_mean']:.4f} (step {row['best_step']}) mwpm {row['mwpm_mean']:.4f}", flush=True)
    except KeyboardInterrupt:
        print(f"\n[exp] interrupted; {len(rows)} finished runs are saved. Continue with:\n"
              f"      python experiment.py --run-dir {run_dir} <same flags>")
    save_rows(run_dir, args, rows)
    print_summary(rows)
    print(f"\n[exp] results.csv / results.json -> {run_dir}")
    return rows


if __name__ == "__main__":
    main()
