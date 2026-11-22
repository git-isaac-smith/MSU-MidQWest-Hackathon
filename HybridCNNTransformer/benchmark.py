#!/usr/bin/env python
"""Time 100 training steps per distance and project the wall-clock cost of 1e7 / 1e8 samples.

    python benchmark.py --config configs/hybrid.yaml
    python benchmark.py --config configs/cnn_baseline.yaml --set compile=true

A step includes Stim sampling, batch construction, forward, backward and the optimiser update.
10 untimed warm-up steps per distance (CUDA/cuDNN autotune, torch.compile) precede the 100 timed ones.
"""
from __future__ import annotations

import argparse
import time

import torch

from qecdec.config import load_config
from qecdec.engine import TrainData, make_optimizer, make_scheduler, train_step
from qecdec.models import build_model
from qecdec.utils import pick_device, seed_everything

STEPS, WARMUP = 100, 10


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def hours(samples: float, samples_per_s: float) -> float:
    return samples / samples_per_s / 3600.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)

    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    model = build_model(cfg, max_d=max(cfg.d_set)).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    run_model = torch.compile(model, dynamic=True) if cfg.compile else model
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg)
    data = TrainData(cfg, device)
    model.train()

    print(f"[bench] model={cfg.model} device={device} batch={cfg.batch_size} bf16={cfg.bf16} "
          f"compile={cfg.compile} p={cfg.p}")
    rate = {}
    for d in cfg.d_set:
        for _ in range(WARMUP):
            train_step(run_model, opt, sched, data.next_batch(d), cfg, device)
        sync(device)
        t_data = 0.0
        t0 = time.perf_counter()
        for _ in range(STEPS):
            ta = time.perf_counter()
            batch = data.next_batch(d)
            t_data += time.perf_counter() - ta
            train_step(run_model, opt, sched, batch, cfg, device)
        sync(device)
        dt = (time.perf_counter() - t0) / STEPS
        rate[d] = cfg.batch_size / dt
        print(f"[bench] d={d}: {dt * 1e3:7.1f} ms/step (data {t_data / STEPS * 1e3:6.1f} ms) "
              f"{rate[d]:9.0f} samples/s | 1e7 -> {hours(1e7, rate[d]):7.2f} h | "
              f"1e8 -> {hours(1e8, rate[d]):8.2f} h")

    # d drawn uniformly from d_set: expected time per sample is the mean of the per-distance times
    mixed = 1.0 / (sum(1.0 / r for r in rate.values()) / len(rate))
    print(f"[bench] mixed d_set={cfg.d_set} (uniform): {mixed:9.0f} samples/s | "
          f"1e7 -> {hours(1e7, mixed):7.2f} h | 1e8 -> {hours(1e8, mixed):8.2f} h")


if __name__ == "__main__":
    main()
