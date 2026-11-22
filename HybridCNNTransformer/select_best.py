#!/usr/bin/env python
"""Pick the best hyperparameters per code distance from experiment.py results.

    python select_best.py                                   # newest folder under experiments/
    python select_best.py experiments/20261122_120000       # a sweep folder (or its results.json)
    python select_best.py experiments/A experiments/B --sigma 1 --metric ler

For every sweep, model and distance d the base d_set of the sweep was validated on, runs are scored on
that distance alone:   score = final validation LER at d / PyMatching's LER on the same shots
(lower is better, 1.0 = as good as MWPM; --metric ler uses the raw LER instead). Then, per distance:

  * baseline = baseline/default + seed/* runs; its spread (std over those runs) is the noise level
  * per hyperparameter group, the best value and its gain over the baseline mean; the gain counts as
    real only if it exceeds --sigma (default 2) noise standard deviations
  * the best single run, and the runs trained on that distance alone (distance/* group)
  * a combined config: the baseline plus every group whose best value was a real gain. Groups ablation,
    steps and shots are reported but never combined (--exclude-groups changes this); p and distance are
    never combined. The combination is NOT tested as a whole: confirm it with a training run.

Results go to  <out-dir>/<YYYYmmdd_HHMMSS>/  (never reused):
    summary.txt                 the report printed to the console
    best_hyperparameters.json   everything above, machine readable
    per_group.csv               one row per (sweep, model, d, group)
    configs/<model>_d<d>.yaml   full-training config for that model and distance (d_set=[d])

Only distances that appear in the sweep's validation results can be ranked. Run experiment.py with
--d-set <d> to tune a distance on its own; the default base d_set 3 5 gives d=3 and d=5.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import yaml

from experiment import CONFIGS, ROOT, SHORT
from qecdec.config import config_to_dict, from_dict, load_config
from qecdec.utils import new_run_dir

REFERENCE = {"baseline", "seed"}        # define the baseline and its noise
NOT_COMPARABLE = {"p"}                  # changes the noise level, scores are not comparable
SEPARATE = {"distance"}                 # changes which distances are trained on; listed separately
DEFAULT_EXCLUDE = ["ablation", "steps", "shots"]    # reported, not combined


# ---------------------------------------------------------------------------------------------------------
def load_sweep(path: str) -> dict:
    f = path if path.endswith(".json") else os.path.join(path, "results.json")
    with open(f) as fh:
        data = json.load(fh)
    rows = [r for r in data["results"] if r["status"] == "ok"]
    return {"name": os.path.basename(os.path.dirname(os.path.abspath(f))), "rows": rows,
            "meta_d": list(data["meta"]["args"]["d_set"])}


def latest_sweep(root: str = "experiments") -> str:
    found = sorted(d for d in (os.listdir(root) if os.path.isdir(root) else [])
                   if os.path.isfile(os.path.join(root, d, "results.json")))
    if not found:
        sys.exit(f"select_best.py: no experiment results under {root}/ (pass a folder or results.json)")
    return os.path.join(root, found[-1])


def score(row: dict, d: int, metric: str):
    v = row["final_by_d"].get(str(d))
    if v is None:
        return None
    if metric == "ler":
        return v
    m = row["mwpm_by_d"].get(str(d), 0)
    return v / m if m > 0 else None        # MWPM made no errors on this validation set: ratio undefined


def label(params: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in params.items())


def analyze(rows: list, model: str, base_d: list, d: int, args) -> dict:
    """All results for one (model, d) of one sweep. `rows` are that sweep's finished runs of `model`."""
    skip = REFERENCE | NOT_COMPARABLE | SEPARATE
    eligible = [r for r in rows if r["d_set"] == base_d and r["group"] not in (NOT_COMPARABLE | SEPARATE)
                and score(r, d, args.metric) is not None]
    ref = [score(r, d, args.metric) for r in eligible if r["group"] in REFERENCE]
    base = {"n": len(ref), "mean": statistics.mean(ref) if ref else None,
            "noise": statistics.stdev(ref) if len(ref) >= 2 else None}
    margin = args.sigma * (base["noise"] or 0.0)

    by_group: dict = {}
    for r in eligible:
        if r["group"] not in REFERENCE:
            by_group.setdefault(r["group"], []).append(r)
    groups = []
    for g, rs in by_group.items():
        best = min(rs, key=lambda r: score(r, d, args.metric))
        s = score(best, d, args.metric)
        gain = None if base["mean"] is None else base["mean"] - s
        verdict = ("no baseline" if gain is None else "better" if gain > margin
                   else "worse" if gain < -margin else "within noise")
        groups.append({"group": g, "best_name": best["name"], "best_params": best["params"], "score": s,
                       "gain": gain, "verdict": verdict, "n_runs": len(rs),
                       "combined": False,
                       "excluded": g in set(args.exclude_groups) or g in skip})
    overrides = {}
    for g in groups:
        if g["verdict"] == "better" and not g["excluded"]:
            g["combined"] = True
            overrides.update(g["best_params"])

    best_run = min(eligible, key=lambda r: score(r, d, args.metric)) if eligible else None
    single = [{"name": r["name"], "d_set": r["d_set"], "score": score(r, d, args.metric)}
              for r in rows if r["group"] in SEPARATE and score(r, d, args.metric) is not None]
    return {"baseline": base, "groups": groups, "overrides": overrides,
            "best_single": None if best_run is None else {
                "name": best_run["name"], "params": best_run["params"], "score": score(best_run, d, args.metric)},
            "distance_runs": single}


def build_config(model: str, d: int, overrides: dict) -> dict:
    raw = config_to_dict(load_config(os.path.join(ROOT, CONFIGS[model])))
    raw.update(overrides)
    raw["d_set"] = [d]
    raw["out_dir"] = f"runs/best_{SHORT[model]}_d{d}"
    return raw


def check_config(raw: dict):
    """(ok, message): the config validates and the model builds for its distance."""
    try:
        cfg = from_dict(raw)
        from qecdec.models import build_model
        build_model(cfg, max(cfg.d_set))
        return True, "builds OK"
    except Exception as e:
        return False, f"INVALID, not written: {e!r}"


# ---------------------------------------------------------------------------------------------------------
def report(sweep: str, model: str, d: int, res: dict, metric: str, cfg_note: str) -> list:
    unit = "final val LER / MWPM LER" if metric == "ratio" else "final val LER"
    b = res["baseline"]
    out = [f"\n=== sweep {sweep} | {model} | d={d} | score = {unit} (lower is better) ==="]
    if b["mean"] is None:
        out.append("baseline: none (run the baseline and seed groups to get a reference and a noise level)")
    else:
        noise = "no noise estimate (need >= 2 baseline/seed runs)" if b["noise"] is None else f"noise std {b['noise']:.4f}"
        out.append(f"baseline: {b['mean']:.4f} over {b['n']} run(s), {noise}")
    out.append(f"  {'group':<16}{'best value':<30}{'score':>9}{'gain':>9}  verdict")
    for g in res["groups"]:
        gain = "n/a" if g["gain"] is None else f"{g['gain']:+.4f}"
        mark = "  (not combined)" if g["excluded"] else "  <- combined" if g["combined"] else ""
        out.append(f"  {g['group']:<16}{label(g['best_params']):<30}{g['score']:>9.4f}{gain:>9}  {g['verdict']}{mark}")
    if res["best_single"]:
        s = res["best_single"]
        out.append(f"best single run: {s['name']} {label(s['params'])} -> {s['score']:.4f}")
    for r in res["distance_runs"]:
        out.append(f"trained on d_set={r['d_set']}: {r['name']} -> {r['score']:.4f}")
    over = label(res["overrides"]) or "(none: no group beat the baseline by more than the noise)"
    out.append(f"combined (untested as a whole): {over}")
    out.append(f"config: {cfg_note}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="*", help="experiment folders or results.json files (default: newest in experiments/)")
    ap.add_argument("--metric", choices=["ratio", "ler"], default="ratio",
                    help="ratio = final val LER / MWPM LER (default), ler = raw final val LER")
    ap.add_argument("--sigma", type=float, default=2.0, help="a gain must exceed this many baseline noise std")
    ap.add_argument("--exclude-groups", nargs="*", default=DEFAULT_EXCLUDE, metavar="GROUP",
                    help=f"groups reported but not combined (default: {' '.join(DEFAULT_EXCLUDE)}); "
                         "pass the flag with no values to combine everything allowed")
    ap.add_argument("--d", type=int, nargs="*", default=None, help="only these distances")
    ap.add_argument("--out-dir", default="best_hyperparameters", help="root folder; each run gets a timestamped subfolder")
    args = ap.parse_args(argv)

    sweeps = [load_sweep(p) for p in (args.results or [latest_sweep()])]
    out_dir = new_run_dir(args.out_dir)
    os.makedirs(os.path.join(out_dir, "configs"))
    lines, summary, csv_rows = [], {}, []
    for sw in sweeps:
        for model in sorted({r["model"] for r in sw["rows"]}):
            rows = [r for r in sw["rows"] if r["model"] == model]
            base_rows = [r for r in rows if r["name"] == "baseline/default"]
            base_d = base_rows[0]["d_set"] if base_rows else sw["meta_d"]
            for d in base_d:
                if args.d and d not in args.d:
                    continue
                res = analyze(rows, model, base_d, d, args)
                if res["baseline"]["mean"] is None and not res["groups"]:
                    continue
                raw = build_config(model, d, res["overrides"])
                ok, msg = check_config(raw)
                cfg_path = os.path.join("configs", f"{SHORT[model]}_d{d}.yaml")
                if ok:
                    with open(os.path.join(out_dir, cfg_path), "w") as f:
                        yaml.safe_dump(raw, f, sort_keys=False)
                res["config"] = {"path": cfg_path if ok else None, "ok": ok, "message": msg, "values": raw}
                summary.setdefault(sw["name"], {}).setdefault(model, {})[str(d)] = res
                lines += report(sw["name"], model, d, res, args.metric, f"{cfg_path if ok else '-'} ({msg})")
                for g in res["groups"]:
                    csv_rows.append({"sweep": sw["name"], "model": model, "d": d, "group": g["group"],
                                     "best_name": g["best_name"], "best_params": json.dumps(g["best_params"]),
                                     "score": g["score"], "baseline_mean": res["baseline"]["mean"],
                                     "noise_std": res["baseline"]["noise"], "gain": g["gain"],
                                     "verdict": g["verdict"], "combined": g["combined"],
                                     "excluded": g["excluded"], "n_runs": g["n_runs"]})
    if not summary:
        lines.append("nothing to analyse: no finished runs with per-distance validation results")
    text = "\n".join(lines).lstrip("\n")
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as f:
        json.dump({"meta": {"args": vars(args), "sweeps": [s["name"] for s in sweeps]}, "results": summary},
                  f, indent=2)
    if csv_rows:
        with open(os.path.join(out_dir, "per_group.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"\n[select] summary.txt, best_hyperparameters.json, per_group.csv, configs/ -> {out_dir}")
    return summary


if __name__ == "__main__":
    main()
