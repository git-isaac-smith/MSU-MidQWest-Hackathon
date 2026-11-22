"""Plots shared by train.py and eval.py (headless matplotlib backend).

    plot_training(run_dir)      loss / error rate / throughput / lr curves from <run_dir>/log.jsonl
    plot_eps_vs_p(path, ...)    per-round logical error rate vs physical error rate, one panel per d

Re-plot a finished (or running) training run from its log:

    python -m qecdec.plotting runs/hybrid                   # writes runs/hybrid/training_curves.png
    python -m qecdec.plotting runs/hybrid --out my.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_PLOT_NAME = "training_curves.png"
EVAL_PLOT_NAME = "eps_vs_p.png"
TIME_PLOT_NAME = "decode_time_vs_d.png"


# ---------------- training curves ----------------
def read_log(path: str):
    """Parse a train.py log.jsonl into (train_records, val_records) of the LAST run in the file.

    The file is opened in append mode, so a re-run into the same out_dir adds a new run after the old one;
    a step number that goes backwards marks the start of a new run and everything before it is dropped.
    """
    train, val, last_step = [], [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step"] < last_step:
                train, val = [], []
            last_step = rec["step"]
            if "val" in rec:
                val.append({"step": rec["step"], **{str(k): v for k, v in rec["val"].items()}})
            else:
                train.append(rec)
    return train, val


def plot_training(run_dir: str, out_path: str | None = None, title: str | None = None) -> str | None:
    """Plot <run_dir>/log.jsonl to <run_dir>/training_curves.png (or out_path). Returns the path, or None if
    the log has no records yet."""
    log_path = os.path.join(run_dir, "log.jsonl")
    train, val = read_log(log_path)
    if not train and not val:
        return None
    out_path = out_path or os.path.join(run_dir, TRAIN_PLOT_NAME)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    (a_loss, a_ler), (a_sps, a_lr) = ax

    if train:
        s = [r["step"] for r in train]
        a_loss.plot(s, [r["loss"] for r in train], color="tab:purple")
        a_ler.plot(s, [r["train_ler"] for r in train], color="0.6", linewidth=1, label="train (running)")
        a_sps.plot(s, [r["samples_per_s"] for r in train], color="tab:green")
        a_lr.plot(s, [r["lr"] for r in train], color="tab:orange")
    a_loss.set_title("training loss"); a_loss.set_yscale("log")
    a_sps.set_title("training throughput (samples/s, validation excluded)")
    a_lr.set_title("learning rate")

    if val:
        s = [r["step"] for r in val]
        for key in sorted((k for k in val[0] if k not in ("step", "mean")), key=int):
            a_ler.plot(s, [r[key] for r in val], marker=".", linewidth=1, label=f"val d={key}")
        a_ler.plot(s, [r["mean"] for r in val], marker="o", color="k", linewidth=1.8, label="val mean")
        best = min(val, key=lambda r: r["mean"])
        for a in (a_loss, a_ler, a_sps, a_lr):
            a.axvline(best["step"], color="k", linestyle=":", linewidth=0.8)
        a_ler.annotate(f"best {best['mean']:.4f} @ {best['step']}", (best["step"], best["mean"]),
                       textcoords="offset points", xytext=(6, 10), fontsize=8)
    a_ler.set_title("logical error rate per shot"); a_ler.set_yscale("log")
    a_ler.legend(fontsize=8)
    for a in (a_loss, a_ler, a_sps, a_lr):
        a.set_xlabel("step"); a.grid(True, which="both", alpha=0.3)
    fig.suptitle(title or os.path.normpath(run_dir), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------- eval: eps vs p ----------------
def plot_eps_vs_p(out_path, names, records, ds, shots):
    fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 4.0), sharey=True, squeeze=False)
    style = {"MWPM": ("k", "o"), "CNN": ("tab:blue", "s"), "Hybrid": ("tab:red", "^")}
    for ax, d in zip(axes[0], ds):
        for name in names:
            rs = sorted((r for r in records if r["d"] == d and r["decoder"] == name), key=lambda r: r["p"])
            color, marker = style.get(name, (None, "o"))
            kw = dict(color=color, marker=marker, capsize=3, linestyle="-", linewidth=1, markersize=5)
            seen = [r for r in rs if r["errors"] > 0]
            if seen:
                ax.errorbar([r["p"] for r in seen], [r["eps"] for r in seen],
                            yerr=[[r["eps"] - r["eps_lo"] for r in seen], [r["eps_hi"] - r["eps"] for r in seen]],
                            label=name, **kw)
            zero = [r for r in rs if r["errors"] == 0]      # no errors seen: show the 95% upper bound
            if zero:
                ax.plot([r["p"] for r in zero], [r["eps_hi"] for r in zero], linestyle="none", marker="v",
                        mfc="none", color=color, label=None if seen else name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"d = {d}" + (" (not trained)" if any(r["outside_train_d"] for r in records if r["d"] == d)
                                  else ""))
        ax.set_xlabel("physical error rate p")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("logical error rate per round  eps")
    axes[0][0].legend()
    fig.suptitle(f"{shots} shots per point, 95% Wilson intervals; hollow v = 0 errors (upper bound)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decode_time(out_path, names, records, ds, batch_size):
    """Decoding time per shot vs d: marker = mean over the p values, bar = min..max over p."""
    style = {"MWPM": ("k", "o"), "CNN": ("tab:blue", "s"), "Hybrid": ("tab:red", "^")}
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name in names:
        xs, mid, lo, hi = [], [], [], []
        for d in ds:
            us = [r["us_per_shot"] for r in records if r["d"] == d and r["decoder"] == name]
            if us:
                xs.append(d); mid.append(sum(us) / len(us)); lo.append(min(us)); hi.append(max(us))
        color, marker = style.get(name, (None, "o"))
        ax.errorbar(xs, mid, yerr=[[m - l for m, l in zip(mid, lo)], [h - m for m, h in zip(mid, hi)]],
                    label=name, color=color, marker=marker, capsize=3, linewidth=1, markersize=5)
    ax.set_yscale("log")
    ax.set_xticks(list(ds))
    ax.set_xlabel("code distance d")
    ax.set_ylabel("decoding time per shot (microseconds)")
    ax.set_title(f"end-to-end decoding time, neural batch {batch_size}", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot training curves from a run directory's log.jsonl")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help=f"output png (default <run_dir>/{TRAIN_PLOT_NAME})")
    args = ap.parse_args(argv)
    path = plot_training(args.run_dir, args.out)
    if path is None:
        ap.error(f"no records in {os.path.join(args.run_dir, 'log.jsonl')}")
    print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
