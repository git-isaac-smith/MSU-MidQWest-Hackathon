"""Fixed-seed comparison against both previous policies on fresh detector records."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from full_map_decoder import FullMapDecoder
from hybrid_policy import HybridDecoder, cnn_priority, decision_rule, gap


BASE = Path(__file__).resolve().parent
MODES = ("mwpm", "balanced", "accuracy", "full_map")
BLOCK = 4096


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_shard(path, **arrays):
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def plan_for(args):
    policy_file = args.policy / "policy.json"
    policy = read_json(policy_file)
    for name, expected in policy["weights_sha256"].items():
        if hashlib.sha256((args.policy / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original policy checkpoint checksum mismatch: {name}")
    return {
        "primary_comparator": "accuracy", "secondary_comparators": ["balanced", "mwpm"],
        "candidate": "All-shot maximum posterior logical parity, ties within 1e-12 use MWPM",
        "physical": policy["physical"], "rounds": 5,
        "policy_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
        "seeds": list(range(args.seed, args.seed + 3)), "shots_per_seed": args.shots_per_seed,
        "sampling_batch": BLOCK, "timing_seed": args.seed + 10, "versions": {
            p: version(p) for p in ("stim", "pymatching", "numpy", "scipy")
        },
    }


def prepare(args, plan):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / "plan.json"
    if lock.exists() and read_json(lock) != plan:
        raise ValueError("Output directory contains a different sample plan")
    if not lock.exists():
        if any(args.out.iterdir()):
            raise FileExistsError("Choose an empty output directory")
        write_json(lock, plan)
    old = HybridDecoder.load(args.policy)
    old.ref.circuit.to_file(args.out / "benchmark.stim")
    for seed in plan["seeds"]:
        dest = args.out / f"sample-{seed}.npz"
        if dest.exists():
            continue
        sampler = old.ref.circuit.compile_detector_sampler(seed=seed)
        blocks = {}
        for i in range(0, plan["shots_per_seed"], BLOCK):
            d, actual = sampler.sample(min(BLOCK, plan["shots_per_seed"] - i), separate_observables=True)
            base = old.ref.mwpm_parity(d)
            costs = old.confidence.features(d)
            idx = np.flatnonzero(gap(costs) <= old.policy["gap_limit"])
            accurate, balanced = base.copy(), base.copy()
            if len(idx):
                exact = old.exact_decisions(d[idx], base[idx])
                accurate[idx] = exact
                score = old.network.predict(old.encode(d[idx], base[idx], costs[idx]))
                balanced[idx] = np.where(cnn_priority(score) >= old.policy["cnn_route_at"], exact,
                                         base[idx] ^ decision_rule(score, old.policy["override_threshold"]))
            batch = dict(detectors=np.packbits(d, axis=1, bitorder="little"), truth=actual[:, 0],
                         mwpm=base, accuracy=accurate, balanced=balanced)
            for name, array in batch.items():
                blocks.setdefault(name, []).append(array)
        save_shard(dest, **{k: np.concatenate(v) for k, v in blocks.items()})
        print(f"Seed {seed}: sampled and decoded {plan['shots_per_seed']:,} shots with previous policies.", flush=True)


def unpack(arr, ref):
    return np.unpackbits(arr, count=ref.circuit.num_detectors, axis=1, bitorder="little").view(np.bool_)


def evaluate(args, plan):
    from full_map_report import summarize, write_report
    model = FullMapDecoder(plan["physical"], args.backend)
    for seed in plan["seeds"]:
        path = args.out / f"posterior-{seed}.npz"
        if path.exists():
            continue
        with np.load(args.out / f"sample-{seed}.npz", allow_pickle=False) as f:
            packed = f["detectors"]
        p = np.empty(len(packed), dtype=np.float64)
        decisions = np.empty(len(packed), dtype=bool)
        for i in range(0, len(packed), BLOCK):
            sl = slice(i, i + BLOCK)
            d = unpack(packed[sl], model.ref)
            p[sl] = model.probabilities(d)
            decisions[sl] = model.decisions(d, p[sl])
        save_shard(path, probability=p, full_map=decisions, backend=args.backend)
        print(f"Seed {seed}: completed all-shot posterior inference.", flush=True)
    if (args.out / "results.json").exists():
        return
    times = measure(args, plan)
    truth, predictions, posterior, ids = collect(args.out, plan)
    summary = summarize(truth, predictions, posterior, ids, times)
    write_report(args.out, summary)
    print([(r["mode"], r["errors"], r["error_percent"]) for r in summary["rows"]], flush=True)


def collect(out, plan):
    truth, probabilities, ids = [], [], []
    preds = {k: [] for k in MODES}
    for seed in plan["seeds"]:
        with np.load(out / f"sample-{seed}.npz") as f, np.load(out / f"posterior-{seed}.npz") as post:
            truth.append(f["truth"])
            probabilities.append(post["probability"])
            for name in MODES:
                preds[name].append(post[name] if name == "full_map" else f[name])
            ids.append(np.full(len(truth[-1]), seed))
    return np.concatenate(truth), {k: np.concatenate(v) for k, v in preds.items()}, np.concatenate(probabilities), np.concatenate(ids)


def measure(args, plan):
    old = HybridDecoder.load(args.policy)
    sampler = old.ref.circuit.compile_detector_sampler(seed=plan["timing_seed"])
    n = min(plan["shots_per_seed"], 16384)
    warm = sampler.sample(min(n, BLOCK))
    # Each timed pass sees previously unseen shots. All policies receive the same three batches.
    batches = [sampler.sample(n) for _ in range(3)]
    start = perf_counter()
    model = FullMapDecoder(plan["physical"], args.backend)
    setup = perf_counter() - start
    methods = {"mwpm": old.ref.mwpm_parity, "balanced": lambda d: old.predict(d, "balanced"),
               "accuracy": lambda d: old.predict(d, "accuracy"), "full_map": model.predict}
    times = {}
    for name, fn in methods.items():
        fn(warm)
        passes = []
        for d in batches:
            start = perf_counter()
            for i in range(0, n, BLOCK):
                fn(d[i:i + BLOCK])
            passes.append(perf_counter() - start)
        times[name] = dict(shots=n, seconds=float(np.median(passes)), batch_size=min(n, BLOCK))
    write_json(args.out / "implementation.json", dict(backend=args.backend, full_map_setup_seconds=setup,
               timing_protocol="Median of three passes over different fresh batches after a separate warm-up.", timings=times))
    return times


def verify(args, plan):
    """Replay Stim, recount every decision and reproduce probabilities independently on a subset."""
    from full_map_report import summarize
    old = HybridDecoder.load(args.policy)
    model = FullMapDecoder(plan["physical"], args.backend)
    reference = FullMapDecoder(plan["physical"], "reference")
    for seed in plan["seeds"]:
        sampler = old.ref.circuit.compile_detector_sampler(seed=seed)
        with np.load(args.out / f"sample-{seed}.npz") as f, np.load(args.out / f"posterior-{seed}.npz") as post:
            packed, truth = f["detectors"], f["truth"]
            decisions, probabilities = post["full_map"], post["probability"]
            for i in range(0, len(truth), BLOCK):
                sl = slice(i, min(i + BLOCK, len(truth)))
                d, a = sampler.sample(sl.stop - i, separate_observables=True)
                np.testing.assert_array_equal(np.packbits(d, axis=1, bitorder="little"), packed[sl])
                np.testing.assert_array_equal(a[:, 0], truth[sl])
                p = model.probabilities(d)
                np.testing.assert_allclose(p, probabilities[sl], rtol=1e-10, atol=1e-12)
                np.testing.assert_array_equal(model.decisions(d, p), decisions[sl])
                if i == 0:
                    np.testing.assert_allclose(reference.probabilities(d[:256]), p[:256], rtol=1e-10, atol=1e-12)
                    np.testing.assert_array_equal(old.predict(d, "accuracy"), f["accuracy"][sl])
                    np.testing.assert_array_equal(old.predict(d, "balanced"), f["balanced"][sl])
    result = read_json(args.out / "results.json")
    truth, preds, posterior, ids = collect(args.out, plan)
    check = summarize(truth, preds, posterior, ids, result["timings"])
    if check["rows"] != result["rows"] or check["paired_comparisons"] != result["paired_comparisons"]:
        raise ValueError("Paired recount disagrees with saved results")
    print(f"Verified {len(truth):,} public records, sampled answers and full-posterior decisions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=BASE / "results" / "hybrid-01")
    parser.add_argument("--stage", choices=("all", "prepare", "score", "verify"), default="all")
    parser.add_argument("--backend", choices=("cached", "native", "reference"), default="cached")
    parser.add_argument("--shots-per-seed", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=82174101)
    args = parser.parse_args()
    if args.shots_per_seed < 2:
        parser.error("Use at least two shots per seed")
    plan = plan_for(args)
    if args.stage in ("prepare", "all"):
        prepare(args, plan)
    if args.stage in ("score", "verify", "all") and read_json(args.out / "plan.json") != plan:
        raise ValueError("The policy or sample plan differs from this run's original plan")
    if args.stage in ("score", "all"):
        evaluate(args, plan)
    if args.stage in ("verify", "all"):
        verify(args, plan)
