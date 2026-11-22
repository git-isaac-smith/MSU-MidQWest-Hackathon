"""Train, lock, and evaluate a PyMatching and Fourier CNN logical decoder."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from frequency import Passband
from hybrid_cnn import HybridCNN, HybridFeatures
from hybrid_data import prepare_split
from hybrid_policy import HybridDecoder, cnn_priority, decision_rule, gap, gap_priority, raw_ablation
from hybrid_report import report
from localizer import Localizer


PHYSICAL = (5, 0.008, 0.04)
GAP_LIMIT = 5.3
EXACT_BUDGET = 0.005
TRAIN_SEED, VALIDATION_SEED = 70731101, 70731102
TEST_SEEDS = (70839111, 70839112, 70839113)
TIMING_SEED = 70731129
CHUNK = 4096
MODES = ("mwpm", "raw_learned", "learned", "margin_only", "balanced", "accuracy")
BASE = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_inputs(decoder, data):
    if not len(data["teacher"]):
        raise ValueError("No hard examples in split. Increase the shot count.")
    return HybridFeatures.concatenate(
        decoder.encode(data["detectors"][i:i + CHUNK], data["baseline"][i:i + CHUNK], data["costs"][i:i + CHUNK])
        for i in range(0, len(data["teacher"]), CHUNK)
    )


def cutoff_from_validation(score, teacher):
    # Expected net repairs are gains minus losses for each proposed change.
    # A confidence threshold may also leave every matching decision alone.
    candidates = [1.0, 0.60, 0.55, 0.525, 0.50, 0.475, 0.45, 0.40]
    return max(candidates, key=lambda t: (2 * teacher[score > t] - 1).sum(dtype=np.float64))


def route_cutoff(priority, total_shots):
    k = max(1, round(EXACT_BUDGET * total_shots))
    return float(np.partition(priority, -min(k, len(priority)))[-min(k, len(priority))])


def read_split(path, expected_shots, expected_seed):
    with np.load(path, allow_pickle=False) as archive:
        physical = tuple(archive[k].item() for k in ("distance", "p_data", "p_readout"))
        if (int(archive["shots"]) != expected_shots or int(archive["seed"]) != expected_seed
                or physical != PHYSICAL or float(archive["gap_limit"]) != GAP_LIMIT
                or bool(archive["train"]) != (expected_seed == TRAIN_SEED)):
            raise ValueError(f"Cached split does not match the requested sample plan: {path}")
        return {key: archive[key] for key in ("detectors", "baseline", "costs", "teacher", "baseline_errors")}


def train(args):
    out = args.out
    if (out / "policy.json").exists():
        raise FileExistsError("A locked policy already exists. Use a fresh output directory to retrain.")
    out.mkdir(parents=True, exist_ok=True)
    for name, size, seed in (("train", args.train_shots, TRAIN_SEED), ("validation", args.validation_shots, VALIDATION_SEED)):
        path = out / f"{name}.npz"
        if not path.exists():
            prepare_split(path, PHYSICAL, size, seed, gap_limit=GAP_LIMIT, train=name == "train")
    source = json.loads((args.site_run / "metrics.json").read_text(encoding="utf-8"))
    if tuple(source["configuration"][k] for k in ("distance", "p_data", "p_readout")) != PHYSICAL:
        raise ValueError("The starting FFT site CNN must match the study's circuit and noise")
    shutil.copyfile(args.site_run / "fft-cnn.npz", out / "site-cnn.npz")
    band = Passband(**source["passband"])
    site_threshold = source["thresholds"]["fft"]
    decoder = HybridDecoder(PHYSICAL, Localizer.load(out / "site-cnn.npz"), site_threshold, band)
    training = read_split(out / "train.npz", args.train_shots, TRAIN_SEED)
    validation = read_split(out / "validation.npz", args.validation_shots, VALIDATION_SEED)
    xt = training_inputs(decoder, training)
    xv = training_inputs(decoder, validation)
    histories, cutoffs = {}, {}
    for name in ("logical-cnn", "raw-cnn"):
        if name == "raw-cnn":
            xt, xv = raw_ablation(xt), raw_ablation(xv)
        model = HybridCNN(xt, seed=77031)
        best = float("inf")
        best_epoch = 0

        def checkpoint(epoch, metrics):
            nonlocal best, best_epoch
            if metrics["validation_loss"] < best:
                best, best_epoch = metrics["validation_loss"], epoch
                model.save(out / f"{name}.npz")
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"{name} epoch {epoch}/{args.epochs}, validation loss {metrics['validation_loss']:.6f}", flush=True)

        history = model.fit(xt, training["teacher"], epochs=args.epochs, batch_size=512, lr=0.001,
                            validation=(xv, validation["teacher"]), progress=checkpoint, seed=119073)
        selected = HybridCNN.load(out / f"{name}.npz").predict(xv)
        cutoffs[name] = cutoff_from_validation(selected, validation["teacher"])
        histories[name] = dict(selected_epoch=best_epoch, epochs=history, override_threshold=cutoffs[name])
        if name == "logical-cnn":
            cnn_route_at = route_cutoff(cnn_priority(selected), args.validation_shots)
    margin_route_at = route_cutoff(gap_priority(validation["detectors"], validation["costs"]), args.validation_shots)
    spec = {
        "physical": PHYSICAL, "rounds": 5, "passband": asdict(band), "site_threshold": site_threshold,
        "gap_limit": GAP_LIMIT, "override_threshold": cutoffs["logical-cnn"], "raw_override_threshold": cutoffs["raw-cnn"],
        "cnn_route_at": cnn_route_at, "gap_route_at": margin_route_at, "target_exact_fraction": EXACT_BUDGET,
        "train_shots": args.train_shots, "train_hard": len(training["teacher"]), "train_seed": TRAIN_SEED,
        "validation_shots": args.validation_shots, "validation_hard": len(validation["teacher"]), "validation_seed": VALIDATION_SEED,
        "test_seeds": TEST_SEEDS, "test_per_seed": args.test_per_seed, "sampling_batch": CHUNK,
        "epochs_per_network": args.epochs, "batch_size_train": 512, "learning_rate": 0.001,
        "init_seed": 77031, "shuffle_seed": 119073,
        "weights_sha256": {n: digest(out / n) for n in ("site-cnn.npz", "logical-cnn.npz", "raw-cnn.npz")},
        "versions": {p: version(p) for p in ("stim", "pymatching", "numpy", "scipy")},
        "input": "Public Z detection events from the included Stim circuit. The sampled logical observable is for external scoring only.",
        "primary": "balanced", "ablation": "raw_learned removes FFT fields, FFT site scores and FFT completion disagreement",
    }
    write_json(out / "training.json", histories)
    decoder.ref.circuit.to_file(out / "benchmark.stim")
    write_json(out / "policy.json", spec)
    print(f"Locked policy after training on {len(training['teacher']):,} hard records and validating on {len(validation['teacher']):,}.", flush=True)


def decisions_for_all(decoder, raw_net, detectors):
    """Share intermediate computations for paired scoring. Timings use independent calls."""
    spec = decoder.policy
    base = decoder.ref.mwpm_parity(detectors)
    costs = decoder.confidence.features(detectors)
    idx = np.flatnonzero(gap(costs) <= spec["gap_limit"])
    decisions = {name: base.copy() for name in MODES}
    routes = {name: np.zeros(len(base), dtype=bool) for name in ("margin_only", "balanced", "accuracy")}
    if len(idx):
        records = detectors[idx]
        inputs = decoder.encode(records, base[idx], costs[idx])
        scores = decoder.network.predict(inputs)
        decisions["learned"][idx] ^= decision_rule(scores, spec["override_threshold"])
        decisions["raw_learned"][idx] ^= decision_rule(raw_net.predict(raw_ablation(inputs)), spec["raw_override_threshold"])
        decisions["balanced"] = decisions["learned"].copy()
        exact = decoder.exact_decisions(records, base[idx])
        masks = {"accuracy": np.ones(len(idx), bool), "balanced": cnn_priority(scores) >= spec["cnn_route_at"],
                 "margin_only": gap_priority(records, costs[idx]) >= spec["gap_route_at"]}
        for name, mask in masks.items():
            routes[name][idx[mask]] = True
            decisions[name][idx[mask]] = exact[mask]
    return decisions, routes


def evaluate(args):
    out = args.out
    if (out / "results.json").exists():
        raise FileExistsError("Completed test results already exist. Use verify to audit them.")
    decoder = HybridDecoder.load(out)
    spec = decoder.policy
    fingerprint = digest(out / "policy.json")
    raw_net = HybridCNN.load(out / "raw-cnn.npz")
    for seed in spec["test_seeds"]:
        dest = out / f"test-{seed}.npz"
        if dest.exists():
            with np.load(dest, allow_pickle=False) as saved:
                if str(saved["policy_sha256"]) != fingerprint or len(saved["truth"]) != spec["test_per_seed"]:
                    raise ValueError("Saved test shard belongs to another policy or sample plan")
            continue
        sampler = decoder.ref.circuit.compile_detector_sampler(seed=seed)
        blocks = {}
        for i in range(0, spec["test_per_seed"], CHUNK):
            d, obs = sampler.sample(min(CHUNK, spec["test_per_seed"] - i), separate_observables=True)
            decisions, routes = decisions_for_all(decoder, raw_net, d)
            batch = {**decisions, **{f"route_{k}": v for k, v in routes.items()},
                     "detectors": np.packbits(d, axis=1, bitorder="little"), "truth": obs[:, 0]}
            for key, arr in batch.items():
                blocks.setdefault(key, []).append(arr)
        staging = dest.with_suffix(".partial.npz")
        np.savez_compressed(staging, policy_sha256=fingerprint, **{k: np.concatenate(v) for k, v in blocks.items()})
        staging.replace(dest)
        print(f"Scored and saved seed {seed}, {spec['test_per_seed']:,} paired shots.", flush=True)

    timings = {}
    timed_shots = min(32768, spec["test_per_seed"])
    d = decoder.ref.circuit.compile_detector_sampler(seed=TIMING_SEED).sample(timed_shots)
    for name in ("mwpm", "learned", "margin_only", "balanced", "accuracy"):
        fn = decoder.ref.mwpm_parity if name == "mwpm" else lambda records: decoder.predict(records, mode=name)
        fn(d[:min(256, len(d))])
        measurements = []
        for _ in range(3):
            start = perf_counter()
            for i in range(0, len(d), CHUNK):
                fn(d[i:i + CHUNK])
            measurements.append(perf_counter() - start)
        timings[name] = dict(shots=len(d), seconds=float(np.median(measurements)), batch_size=CHUNK)
    truth, preds, routes, seeds = collect_saved(out, spec)
    report(out, truth, preds, routes, seeds, timings)


def collect_saved(out, spec):
    archive = {}
    seeds = []
    for seed in spec["test_seeds"]:
        with np.load(out / f"test-{seed}.npz", allow_pickle=False) as f:
            for k in f.files:
                if k != "detectors" and k != "policy_sha256":
                    archive.setdefault(k, []).append(f[k])
            seeds.append(np.full(len(f["truth"]), seed, dtype=np.int64))
    archive = {k: np.concatenate(v) for k, v in archive.items()}
    return (archive["truth"], {k: archive[k] for k in MODES},
            {k: archive[f"route_{k}"] for k in ("margin_only", "balanced", "accuracy")}, np.concatenate(seeds))


def verify(out):
    decoder = HybridDecoder.load(out)
    spec = decoder.policy
    raw_net = HybridCNN.load(out / "raw-cnn.npz")
    for name, expected in spec["weights_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Checkpoint differs from locked policy: {name}")
    for seed in spec["test_seeds"]:
        sampler = decoder.ref.circuit.compile_detector_sampler(seed=seed)
        with np.load(out / f"test-{seed}.npz", allow_pickle=False) as f:
            truth, packed = f["truth"], f["detectors"]
            saved_primary = f["balanced"]
            for i in range(0, len(truth), CHUNK):
                stop = min(i + CHUNK, len(truth))
                d, obs = sampler.sample(stop - i, separate_observables=True)
                np.testing.assert_array_equal(np.packbits(d, axis=1, bitorder="little"), packed[i:stop])
                np.testing.assert_array_equal(obs[:, 0], truth[i:stop])
                np.testing.assert_array_equal(decoder.predict(d, "balanced"), saved_primary[i:stop])
                if i == 0:
                    for mode in ("learned", "accuracy", "margin_only"):
                        np.testing.assert_array_equal(decoder.predict(d, mode), f[mode][:stop])
                    predictions, _ = decisions_for_all(decoder, raw_net, d)
                    for mode in MODES:
                        np.testing.assert_array_equal(predictions[mode], f[mode][:stop])
    truth, preds, _, _ = collect_saved(out, spec)
    checked = set()
    with (out / "results.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["scope"] != "pooled":
                continue
            mode = row["mode"]
            n = int(np.count_nonzero(truth != preds[mode]))
            if n != int(row["errors"]) or len(truth) != int(row["total"]):
                raise ValueError(f"Recount differs for {mode}")
            checked.add(mode)
    if checked != set(MODES):
        raise ValueError("Summary is missing a decoder")
    print(f"Replayed all {len(truth):,} detector records, logical observables and deployed hybrid predictions; recounted every method.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site-run", type=Path, default=BASE / "results" / "screen-run-2026112102")
    parser.add_argument("--stage", choices=("all", "fit", "test", "verify"), default="all")
    parser.add_argument("--train-shots", type=int, default=2_400_000)
    parser.add_argument("--validation-shots", type=int, default=300_000)
    parser.add_argument("--test-per-seed", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if min(args.train_shots, args.validation_shots, args.test_per_seed, args.epochs) < 1:
        parser.error("Sample sizes and epochs must be positive")
    if args.stage in ("fit", "all"):
        train(args)
    if args.stage in ("test", "all"):
        evaluate(args)
    if args.stage in ("verify", "all"):
        verify(args.out)
