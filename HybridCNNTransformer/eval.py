#!/usr/bin/env python
"""Compare MWPM, the CNN baseline and the hybrid decoder on identical fresh Stim shots.

    python eval.py                      # MWPM + CNN + hybrid from the default checkpoints
    python eval.py --hybrid runs/hybrid/best.pt --cnn runs/cnn_baseline/best.pt
    python eval.py --hybrid runs/hybrid/best.pt --shots 50000 --d 3 5 7 9 --p 0.001 0.003

Per-shot logical error rate P_L and per-round rate
    eps = (1 - (1 - 2 P_L)^(1 / rounds)) / 2,   rounds = d,
with 95% Wilson intervals (per-round bars are the interval endpoints mapped through P_L -> eps).

Decoding time is measured for every decoder on the same shots, end to end from the detection events to
the predicted observable flip, and reported as microseconds per shot (decode_s, us_per_shot, shots_per_s):
    MWPM    pymatching decode_batch on all shots (single thread); the one-off matching-graph build is
            reported separately as setup_s
    CNN/Hybrid   batch building (grid / token features) + forward pass + copy back, at --batch-size, after a
            warm-up batch (not timed) so one-off initialisation is excluded
Timings depend on --batch-size, --device, the torch thread count and other load on the machine; run on an
idle machine and compare decoders within one run.

Every run writes to its own timestamped folder  <out-dir>/<YYYYmmdd_HHMMSS>/  (default out-dir
eval_results); an existing folder is never reused, so runs never overwrite each other:
    results.csv   one row per (d, p, decoder)
    results.json  the same rows plus arguments and checkpoint info
    eps_vs_p.png  per-round error rate vs p, one panel per d
    decode_time_vs_d.png  decoding time per shot vs d, one line per decoder
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch

from qecdec.config import from_dict
from qecdec.data import GeometryCache, ShotSource, build_mwpm, mwpm_predict
from qecdec.engine import neural_predict
from qecdec.models import build_model
from qecdec.plotting import EVAL_PLOT_NAME, TIME_PLOT_NAME, plot_decode_time, plot_eps_vs_p
from qecdec.utils import eps_round, new_run_dir, pick_device, shot_and_round_rates, wilson_interval

DEFAULT_P = [0.001, 0.002, 0.003, 0.005, 0.007]
DEFAULT_D = [3, 5, 7, 9]
DEFAULT_CKPT = {"CNN": "runs/cnn_baseline/best.pt", "Hybrid": "runs/hybrid/best.pt"}


def load_neural(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = from_dict(ck["cfg"])
    model = build_model(cfg, max_d=max(cfg.d_set)).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    model.load_state_dict(ck["model"])
    model.eval()
    return cfg, model, {"path": path, "step": ck.get("step"), "train_p": cfg.p, "train_d_set": cfg.d_set,
                        "model": cfg.model}


def make_record(d, p, name, k, n, outside, decode_s, setup_s=0.0):
    P, P_err, eps, eps_err = shot_and_round_rates(k, n, d)
    lo, hi = wilson_interval(k, n)
    return {"d": d, "p": p, "decoder": name, "shots": n, "errors": k,
            "P_L": P, "P_L_err": P_err, "P_L_lo": lo, "P_L_hi": hi,
            "eps": eps, "eps_err": eps_err, "eps_lo": eps_round(lo, d), "eps_hi": eps_round(hi, d),
            "decode_s": decode_s, "us_per_shot": 1e6 * decode_s / n, "shots_per_s": n / decode_s,
            "setup_s": setup_s, "outside_train_d": outside}


def print_table(title, names, records, kind):
    cw = 22
    mwpm_us = {(r["d"], r["p"]): r["us_per_shot"] for r in records if r["decoder"] == "MWPM"}

    def cell(r):
        if kind == "shot":
            return f"{r['P_L']:.4f}+/-{r['P_L_err']:.4f}"
        if kind == "time":
            s = f"{r['us_per_shot']:.1f} us"
            return s if r["decoder"] == "MWPM" else f"{s} ({r['us_per_shot'] / mwpm_us[(r['d'], r['p'])]:.2f}x)"
        return f"{r['eps']:.2e}+/-{r['eps_err']:.1e}"

    cells = {(r["d"], r["p"], r["decoder"]): (cell(r), r["outside_train_d"]) for r in records}
    print(f"\n{title}")
    print(f"{'d':>3} {'p':>6} | " + " | ".join(f"{n:^{cw}}" for n in names))
    print("-" * (13 + (cw + 3) * len(names)))
    seen = []
    for r in records:
        if (r["d"], r["p"]) in seen:
            continue
        seen.append((r["d"], r["p"]))
        d, p = seen[-1]
        mark = "*" if r["outside_train_d"] else ""
        print(f"{str(d) + mark:>3} {p:>6.3f} | " + " | ".join(f"{cells[(d, p, n)][0]:^{cw}}" for n in names))


def save_results(run_dir, args, ckpts, records):
    fields = list(records[0].keys())
    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    meta = {"timestamp": os.path.basename(run_dir), "args": vars(args), "checkpoints": ckpts,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": records}, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", default=None,
                    help=f"hybrid checkpoint (default {DEFAULT_CKPT['Hybrid']} if it exists)")
    ap.add_argument("--cnn", default=None,
                    help=f"CNN baseline checkpoint (default {DEFAULT_CKPT['CNN']} if it exists)")
    ap.add_argument("--shots", type=int, default=20000, help="shots per (d, p) cell")
    ap.add_argument("--p", type=float, nargs="*", default=DEFAULT_P)
    ap.add_argument("--d", type=int, nargs="*", default=DEFAULT_D)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="eval_results", help="root folder; each run gets a timestamped subfolder")
    ap.add_argument("--no-plot", action="store_true", help="skip eps_vs_p.png and decode_time_vs_d.png")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    neural, ckpts = {}, {}
    for name, given in ("CNN", args.cnn), ("Hybrid", args.hybrid):
        path = given or DEFAULT_CKPT[name]
        if not os.path.isfile(path):
            if given:
                ap.error(f"{name} checkpoint not found: {path}")
            print(f"[eval] WARNING: no {name} checkpoint at {path}; {name} will be missing from the "
                  f"tables and plot (train it first, or pass --{name.lower()} PATH)")
            continue
        cfg, model, info = load_neural(path, device)
        neural[name], ckpts[name] = (cfg, model), info
    names = ["MWPM"] + list(neural)
    geoms = {n: GeometryCache(cfg.boundary_clip, device) for n, (cfg, _) in neural.items()}
    trained_d = set().union(*[set(cfg.d_set) for cfg, _ in neural.values()]) if neural else set()

    records, warm = [], set()
    for d in args.d:
        outside = bool(trained_d) and d not in trained_d
        for pi, p in enumerate(args.p):
            t0 = time.perf_counter()
            src = ShotSource(d, p, seed=900_000_000 + 1000 * d + pi)     # same shots for every decoder
            det, obs = src.sample(args.shots)
            t = time.perf_counter()
            matching = build_mwpm(src.circuit)
            setup = {"MWPM": time.perf_counter() - t}                      # one-off graph build, not decode time
            preds, secs = {}, {}
            t = time.perf_counter()
            preds["MWPM"] = mwpm_predict(matching, det)
            secs["MWPM"] = time.perf_counter() - t
            for n, (cfg, model) in neural.items():
                if (n, d) not in warm:      # untimed warm-up: kernel/primitive setup, geometry cache for this d
                    neural_predict(model, det[:args.batch_size], obs[:args.batch_size], d, geoms[n], device,
                                   args.batch_size, cfg.bf16, cfg.channels_last)
                    warm.add((n, d))
                t = time.perf_counter()
                preds[n] = neural_predict(model, det, obs, d, geoms[n], device, args.batch_size,
                                          cfg.bf16, cfg.channels_last)
                secs[n] = time.perf_counter() - t
            for n, pr in preds.items():
                records.append(make_record(d, p, n, int((pr != obs).sum()), len(obs), outside, secs[n],
                                           setup.get(n, 0.0)))
            print(f"[eval] d={d} p={p} done in {time.perf_counter() - t0:.1f}s", flush=True)

    print_table("Logical error rate per shot  (P_L +/- 95% Wilson half-width)", names, records, "shot")
    print_table("Logical error rate per round (eps +/- 95% half-width)", names, records, "round")
    print_table(f"Decoding time per shot (end to end, batch {args.batch_size}; neural in brackets: x MWPM time)",
                names, records, "time")
    print(f"\n[eval] timing: device={device}, torch threads={torch.get_num_threads()}; MWPM is single-threaded "
          f"(its graph build is in setup_s, not counted)")
    if any(r["outside_train_d"] for r in records):
        print("\n* d outside the training d_set of the neural models (generalisation test)")

    run_dir = new_run_dir(args.out_dir)
    save_results(run_dir, args, ckpts, records)
    files = "results.csv, results.json"
    if not args.no_plot:
        plot_eps_vs_p(os.path.join(run_dir, EVAL_PLOT_NAME), names, records, args.d, args.shots)
        plot_decode_time(os.path.join(run_dir, TIME_PLOT_NAME), names, records, args.d, args.batch_size)
        files += f", {EVAL_PLOT_NAME}, {TIME_PLOT_NAME}"
    print(f"\n[eval] saved {files} -> {run_dir}")


if __name__ == "__main__":
    main()
