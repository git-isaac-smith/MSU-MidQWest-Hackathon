#!/usr/bin/env python
"""Train the hybrid decoder or the CNN baseline on on-the-fly Stim samples.

    python train.py --config configs/hybrid.yaml
    python train.py --config configs/hybrid.yaml --set steps=5000 p=0.004 use_boundary_tokens=false
    python train.py --config configs/cnn_baseline.yaml --no-plot

Training curves (loss, train/val error rate per d, throughput, lr) are written to
<out_dir>/training_curves.png at every validation and at the end. To re-plot an existing run:
    python -m qecdec.plotting runs/hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from qecdec.config import config_to_dict, load_config
from qecdec.data import GeometryCache
from qecdec.engine import (TrainData, make_optimizer, make_scheduler, make_val_sets,
                           neural_predict, train_step)
from qecdec.models import build_model
from qecdec.plotting import plot_training
from qecdec.utils import pick_device, seed_everything


def validate(model, cfg, val_sets, geom, device) -> dict:
    res = {}
    for d, (det, obs) in val_sets.items():
        pred = neural_predict(model, det, obs, d, geom, device, 1024, cfg.bf16, cfg.channels_last)
        res[d] = float((pred != obs).mean())
    res["mean"] = sum(res[d] for d in val_sets) / len(val_sets)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. steps=2000 p=0.004")
    ap.add_argument("--no-plot", action="store_true", help="do not write <out_dir>/training_curves.png")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    return run(cfg, plot=not args.no_plot)


def _plot(cfg, log_f) -> None:
    """Refresh the training-curve plot; a plotting problem must never kill a long training run."""
    log_f.flush()
    try:
        plot_training(cfg.out_dir)
    except Exception as e:
        print(f"[train] WARNING: could not write training plot: {e!r}")


def run(cfg, plot: bool = True) -> dict:
    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    model = build_model(cfg, max_d=max(cfg.d_set)).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model={cfg.model} params={n_params/1e3:.1f}k device={device} p={cfg.p} d_set={cfg.d_set}")
    run_model = torch.compile(model, dynamic=True) if cfg.compile else model

    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg)
    data = TrainData(cfg, device)
    geom = GeometryCache(cfg.boundary_clip, device)
    val_sets = make_val_sets(cfg)

    best = float("inf")
    log_f = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
    acc_loss = torch.zeros((), device=device)
    acc_ler = torch.zeros((), device=device)
    n_acc, samples, last_val, t_acc = 0, 0, {}, 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_log = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        batch = data.next_batch()
        loss, ler = train_step(run_model, opt, sched, batch, cfg, device)
        acc_loss += loss
        acc_ler += ler
        n_acc += 1
        samples += cfg.batch_size

        if step % cfg.log_every == 0 or step == cfg.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = t_acc + (time.perf_counter() - t_log)
            rec = {"step": step, "loss": float(acc_loss / n_acc), "train_ler": float(acc_ler / n_acc),
                   "samples_per_s": samples / dt, "lr": sched.get_last_lr()[0]}
            print(f"[train] step {step:6d} loss {rec['loss']:.4f} ler {rec['train_ler']:.4f} "
                  f"{rec['samples_per_s']:.0f} samples/s lr {rec['lr']:.2e}")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            acc_loss.zero_(); acc_ler.zero_(); n_acc, samples, t_acc = 0, 0, 0.0
            t_log = time.perf_counter()

        if step % cfg.val_every == 0 or step == cfg.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_acc += time.perf_counter() - t_log   # training time so far; validation is excluded
            last_val = validate(model, cfg, val_sets, geom, device)
            print(f"[val]   step {step:6d} " + " ".join(f"d{d}={v:.4f}" for d, v in last_val.items()))
            log_f.write(json.dumps({"step": step, "val": last_val}) + "\n"); log_f.flush()
            ck = {"cfg": config_to_dict(cfg), "model": model.state_dict(), "step": step, "val": last_val}
            torch.save(ck, os.path.join(cfg.out_dir, "last.pt"))
            if last_val["mean"] < best:
                best = last_val["mean"]
                torch.save(ck, os.path.join(cfg.out_dir, "best.pt"))
                print(f"[val]   new best mean LER {best:.4f} -> best.pt")
            if plot:
                _plot(cfg, log_f)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_log = time.perf_counter()
    if plot:      # already refreshed at the final validation (step == cfg.steps)
        print(f"[train] training curves -> {os.path.join(cfg.out_dir, 'training_curves.png')}")
    log_f.close()
    return {"best_val_mean": best, "last_val": last_val}


if __name__ == "__main__":
    main()
