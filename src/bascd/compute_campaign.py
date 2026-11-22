"""Train, score and time the distance-5/7 decoder compute study."""

import argparse
import csv
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pymatching
import stim
from scipy.stats import binomtest

from .batch_workers import ParallelBatchDecoder
from .capped_gap import CappedGapProbe, CappedGapRoutedDecoder
from .circuits import memory_circuit
from .css_routed import CssRoutedDecoder, _counts
from .disagreement_rescue import DisagreementRescue
from .learned_fault_router import LearnedFaultRouter
from .projected_router import ProjectedOrdinary
from .stats import paired_ler_difference, wilson95

PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)
METHODS = (
    "ordinary",
    "activity",
    "correlated",
    "learned",
    "learned_rescue",
    "gap2",
    "disagreement_rescue",
    "belief",
)
CORE = ("ordinary", "activity", "correlated", "learned", "disagreement_rescue")
DISTANCES = (5, 7)
SOURCES = (
    "compute_campaign",
    "learned_fault_router",
    "projected_router",
    "capped_gap",
    "css_routed",
    "batch_workers",
    "circuits",
    "stats",
    "fast_hybrid",
    "disagreement_rescue",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    with path.open("x", encoding="utf8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def write_csv(path, rows):
    with path.open("x", encoding="utf8", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=rows[0].keys())
        out.writeheader()
        out.writerows(rows)


def plan(smoke):
    # The smoke harness and the development pilots must never expose a confirmatory shot.
    offset = 20000000 if smoke else 0
    return {
        "model": "provisional simple_uniform, rotated Z memory, p=.005, rounds=distance",
        "methods": METHODS,
        "distances": DISTANCES,
        "activity_calibration_shots": 8192,
        "activity_fraction": 0.06,
        "learned_fraction": 0.052,
        "learned_temporal": True,
        "learned_parity_distances": [5],
        "gap_cutoff": 2.0,
        "belief_iterations": 12,
        "training_shots_per_seed": 2000 if smoke else 400000,
        "test_shots_per_seed": 257 if smoke else 100000,
        "test_seeds_per_distance": 3,
        "training_seeds": {str(d): [91500000 + offset + d * 100 + i for i in range(3)] for d in DISTANCES},
        "test_seeds": {str(d): [91600000 + offset + d * 100 + i for i in range(3)] for d in DISTANCES},
        "activity_seeds": {str(d): 91700000 + offset + d for d in DISTANCES},
        "random_control_seed_offset": 1000000,
        "timing_seed_base": 91900000 + offset,
        "primary_accuracy": "learned vs activity and disagreement_rescue vs correlated at d5,d7; "
        "two-sided exact McNemar, Bonferroni alpha .0125 for four comparisons",
        "other_comparisons": "descriptive; random_control routes exactly the same number of records "
        "per seed as learned, uniformly without replacement",
        "scoring": "reuse identical detector-only branch outputs for scoring; independently verify "
        "assembled decisions against deployed decoders on the first 256 records of each seed",
        "timing": "independent fresh paired records; warmed, interleaved randomized single-process "
        "calls; separate persistent-process IPC scaling includes transfers",
        "packages": {s: version(s) for s in ("stim", "numpy", "pymatching", "scipy", "stabilizer-qec")},
        "host": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "source_sha256": {n: digest(Path(__file__).with_name(n + ".py")) for n in SOURCES},
        "smoke": smoke,
    }


def prepare(folder, smoke):
    folder.mkdir(parents=True, exist_ok=False)
    protocol = plan(smoke)
    write_json(folder / "protocol.json", protocol)
    for d in DISTANCES:
        case = folder / f"d{d}"
        case.mkdir()
        circuit = memory_circuit("z", d, d, 0.005)
        circuit.to_file(case / "circuit.stim")
        model = circuit.detector_error_model(decompose_errors=True)
        t = perf_counter()
        policy = LearnedFaultRouter.fit_sampled(
            circuit,
            model,
            seeds=protocol["training_seeds"][str(d)],
            shots_per_seed=protocol["training_shots_per_seed"],
            fraction=0.052,
            temporal=True,
            parity=(d == 5),
        )
        policy.save(case / "policy.npz")
        write_json(
            case / "freeze.json",
            dict(
                distance=d,
                circuit_sha256=digest(case / "circuit.stim"),
                policy_sha256=digest(case / "policy.npz"),
                training_seconds=perf_counter() - t,
                learned_strict_cutoff=float(policy.cutoff),
                policy_bytes=(case / "policy.npz").stat().st_size,
            ),
        )
        print(f"Frozen d={d} model at {case}", flush=True)


def _belief(model):
    from stabilizer_qec import BeliefMatching

    return BeliefMatching(model, max_bp_iters=12, bp_method="product_sum")


class SpecialistCascade:
    """Use a frozen learned gate and optionally resolve its experts' disagreements."""

    def __init__(self, circuit, model, policy, rescue=False, threads=1):
        self.ordinary = ProjectedOrdinary(circuit, model)
        self.correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.route = policy.route_batch
        self.belief = _belief(model) if rescue else None
        self.threads = threads

    def decode_batch(self, x):
        route = self.route(x)
        if self.belief is None:
            # The cheap branch can skip the escalated shots entirely.
            pred = np.empty((len(x), 1), np.uint8)
            if np.any(~route):
                pred[~route] = self.ordinary.decode_batch(x[~route])
        else:
            pred = self.ordinary.decode_batch(x)
        if np.any(route):
            corr = self.correlated.decode_batch(x[route], enable_correlations=True, **PACKED)
            if self.belief is not None:
                disputed = np.flatnonzero((pred[route] != corr)[:, 0])
                if len(disputed):
                    disputed_rows = np.flatnonzero(route)[disputed]
                    corr[disputed] = self.belief.decode_batch(
                        x[disputed_rows], threads=self.threads, **PACKED
                    )
            pred[route] = corr
        return pred


def factory(mode, case_path, threads=1):
    case = Path(case_path)
    circuit = stim.Circuit.from_file(case / "circuit.stim")
    model = circuit.detector_error_model(decompose_errors=True)
    if mode in ("ordinary", "correlated"):
        corr = mode == "correlated"
        matcher = pymatching.Matching.from_detector_error_model(model, enable_correlations=corr)
        return partial(matcher.decode_batch, enable_correlations=corr, **PACKED)
    if mode == "belief":
        return partial(_belief(model).decode_batch, threads=threads, **PACKED)
    if mode == "activity":
        d = json.loads((case / "freeze.json").read_text())["distance"]
        calibration = json.loads((case.parent / "protocol.json").read_text())["activity_seeds"][str(d)]
        return CssRoutedDecoder(circuit, model, calibration_seed=calibration).decode_batch
    if mode == "gap2":
        return CappedGapRoutedDecoder(circuit, model, 2.0).decode_batch
    if mode == "disagreement_rescue":
        return DisagreementRescue(
            model, ordinary=ProjectedOrdinary(circuit, model).decode_batch, threads=threads
        ).decode_batch
    if mode not in ("learned", "learned_rescue"):
        raise ValueError("unknown method")
    policy = LearnedFaultRouter.load(case / "policy.npz")
    if policy.model_digest != sha256(str(model).encode()).hexdigest():
        raise ValueError("routing policy and circuit model differ")
    return SpecialistCascade(
        circuit, model, policy, rescue=(mode == "learned_rescue"), threads=threads
    ).decode_batch


def failure_summary(bits, truth):
    n = len(truth)
    errors = int(np.count_nonzero(bits != truth))
    lo, hi = wilson95(errors, n)
    return dict(
        shots=n,
        errors=errors,
        error_percent=100 * errors / n,
        wilson_low_percent=100 * lo,
        wilson_high_percent=100 * hi,
    )


def score(folder, protocol):
    summary, pairs = [], []
    for d in DISTANCES:
        case = folder / f"d{d}"
        frozen = json.loads((case / "freeze.json").read_text())
        if any(
            digest(case / file) != frozen[tag]
            for file, tag in (("policy.npz", "policy_sha256"), ("circuit.stim", "circuit_sha256"))
        ):
            raise ValueError("frozen files have changed")
        seeds = protocol["test_seeds"][str(d)]
        circuit = stim.Circuit.from_file(case / "circuit.stim")
        model = circuit.detector_error_model(decompose_errors=True)
        ordinary = factory("ordinary", case)
        correlated = factory("correlated", case)
        projected = ProjectedOrdinary(circuit, model)
        policy = LearnedFaultRouter.load(case / "policy.npz")
        activity = CssRoutedDecoder(circuit, model, calibration_seed=protocol["activity_seeds"][str(d)])
        gap = CappedGapProbe(model, 2.0)
        belief = factory("belief", case, threads=6)
        verify = {m: factory(m, case) for m in METHODS}
        all_rows = []
        route_stats = []
        for seed in seeds:
            n = protocol["test_shots_per_seed"]
            packed, observables = circuit.compile_detector_sampler(seed=seed).sample(
                n, separate_observables=True, bit_packed=True
            )
            truth = observables[:, 0]
            pred = {name: np.empty(n, dtype=np.uint8) for name in (*METHODS, "random_control")}
            learned_mask = policy.route_batch(packed)
            random_mask = np.zeros(n, dtype=bool)
            rng = np.random.default_rng(seed + protocol["random_control_seed_offset"])
            random_mask[rng.choice(n, int(learned_mask.sum()), replace=False)] = True
            totals = dict(learned=0, activity=0, gap2=0, disagreement_rescue=0, learned_rescue=0)
            for start in range(0, n, 4096):
                s = slice(start, start + 4096)
                x = packed[s]
                easy = projected.decode_batch(x)
                corr = correlated(x)[:, 0]
                bp = belief(x)[:, 0]
                act = _counts(x) > activity.cutoff
                _, near = gap.classify(x, easy)
                easy = easy[:, 0]
                learned = learned_mask[s]
                disputed = easy != corr
                for k, mask in (
                    ("learned", learned),
                    ("activity", act),
                    ("gap2", near),
                    ("disagreement_rescue", disputed),
                    ("learned_rescue", learned & disputed),
                ):
                    totals[k] += int(mask.sum())
                pred["ordinary"][s] = ordinary(x)[:, 0]
                pred["correlated"][s] = corr
                pred["belief"][s] = bp
                pred["activity"][s] = np.where(act, corr, easy)
                pred["learned"][s] = np.where(learned, corr, easy)
                pred["gap2"][s] = np.where(near, corr, easy)
                rescue = np.where(disputed, bp, corr)
                pred["disagreement_rescue"][s] = rescue
                pred["learned_rescue"][s] = np.where(learned, rescue, easy)
                pred["random_control"][s] = np.where(random_mask[s], corr, easy)
            for name, fn in verify.items():
                if not np.array_equal(fn(packed[:256])[:, 0], pred[name][:256]):
                    raise ValueError(f"deployment mismatch for {name} seed {seed}")
            with (case / f"test-{seed}.npz").open("xb") as out:
                np.savez_compressed(
                    out,
                    detectors=packed,
                    truth=truth,
                    **{k: np.packbits(v, bitorder="little") for k, v in pred.items()},
                )
            all_rows.append(({k: v != truth for k, v in pred.items()}, n))
            route_stats.append(dict(seed=seed, shots=n, activity_cutoff=activity.cutoff, **totals))
            print(f"d={d} verified held-out seed {seed}", flush=True)
        write_csv(case / "route-counts.csv", route_stats)
        for name in (*METHODS, "random_control"):
            errors = np.concatenate([out[name] for out, n in all_rows])
            summary.append(dict(distance=d, method=name, **failure_summary(errors, np.zeros_like(errors))))
            for base in ("ordinary", "activity", "correlated", "belief"):
                wins = [int((out[base] & ~out[name]).sum()) for out, n in all_rows]
                losses = [int((~out[base] & out[name]).sum()) for out, n in all_rows]
                nshots = [n for _, n in all_rows]
                result = paired_ler_difference(nshots, wins, losses)
                w, l = sum(wins), sum(losses)
                ci = result["approximate_95_ci"] or [None, None]
                pairs.append(
                    dict(
                        distance=d,
                        method=name,
                        baseline=base,
                        recovered=w,
                        introduced=l,
                        change_pp=100 * result["ler_difference"],
                        ci_low_pp=100 * ci[0] if ci[0] is not None else None,
                        ci_high_pp=100 * ci[1] if ci[1] is not None else None,
                        paired_p=binomtest(w, w + l, 0.5).pvalue if w + l else 1.0,
                    )
                )
    write_csv(folder / "accuracy.csv", summary)
    write_csv(folder / "paired.csv", pairs)


def timed(fn, x):
    start = perf_counter()
    bits = fn(x)
    return bits, (perf_counter() - start) * 1e6 / len(x)


def time_serial(folder, protocol):
    rows = []
    setup = []
    for d in DISTANCES:
        case = folder / f"d{d}"
        fns = {}
        for name in METHODS:
            t = perf_counter()
            fns[name] = factory(name, case)
            setup.append(dict(distance=d, method=name, setup_s=perf_counter() - t))
        seed = protocol["timing_seed_base"] + d
        circuit = stim.Circuit.from_file(case / "circuit.stim")
        sampler = circuit.compile_detector_sampler(seed=seed)
        warm = sampler.sample(256, bit_packed=True)
        for fn in fns.values():
            fn(warm)
        rng = np.random.default_rng(seed)
        shots = 64 if protocol["smoke"] else 2048
        for i in range(2 if protocol["smoke"] else 11):
            x = sampler.sample(shots, bit_packed=True)
            for name in rng.permutation(tuple(fns)):
                _, cost = timed(fns[name], x)
                rows.append(
                    dict(
                        distance=d,
                        method=name,
                        measurement="batch",
                        repetition=i,
                        shots=shots,
                        microseconds_per_shot=cost,
                    )
                )
        singles = sampler.sample(8 if protocol["smoke"] else 96, bit_packed=True)
        for i in range(len(singles)):
            for name in rng.permutation(tuple(fns)):
                _, cost = timed(fns[name], singles[i : i + 1])
                rows.append(
                    dict(
                        distance=d,
                        method=name,
                        measurement="single",
                        repetition=i,
                        shots=1,
                        microseconds_per_shot=cost,
                    )
                )
        print(f"Finished d={d} isolated serial timing", flush=True)
    write_csv(folder / "timing.csv", rows)
    write_csv(folder / "setup.csv", setup)


def scaling(folder, protocol):
    d = 7
    case = folder / f"d{d}"
    seed = protocol["timing_seed_base"] + 100 + d
    rng = np.random.default_rng(seed)
    circuit = stim.Circuit.from_file(case / "circuit.stim")
    sampler = circuit.compile_detector_sampler(seed=seed)
    n = 128 if protocol["smoke"] else 8192
    warm = sampler.sample(n, bit_packed=True)
    batches = [sampler.sample(n, bit_packed=True) for _ in range(2 if protocol["smoke"] else 9)]
    methods = ("ordinary", "correlated", "learned", "disagreement_rescue")
    refs = {m: [factory_fn(x) for x in batches] for m in methods for factory_fn in [factory(m, case)]}
    jobs = [(m, w) for m in methods for w in ((1, 2) if protocol["smoke"] else (1, 2, 4, 8))]
    rows = []
    for index in rng.permutation(len(jobs)):
        m, w = jobs[index]
        with ParallelBatchDecoder(factory, (m, str(case.resolve())), workers=w) as pool:
            pool.warmup(warm)
            for i, x in enumerate(batches):
                bits, cost = timed(pool.decode_batch, x)
                if not np.array_equal(bits, refs[m][i]):
                    raise ValueError(f"worker mismatch: {m}, {w} workers")
                rows.append(
                    dict(
                        distance=d,
                        method=m,
                        workers=w,
                        shots=n,
                        repetition=i,
                        microseconds_per_shot=cost,
                        setup_s=pool.startup_seconds,
                        warmup_s=pool.warmup_seconds,
                    )
                )
        print(f"Verified d={d} {m} at {w} worker(s)", flush=True)
    write_csv(folder / "scaling.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "score", "time", "scale"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--smoke", action="store_true", help="small prepare run; later stages inherit its protocol"
    )
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.out, args.smoke)
    else:
        protocol = json.loads((args.out / "protocol.json").read_text())
        if any(
            digest(Path(__file__).with_name(n + ".py")) != h for n, h in protocol["source_sha256"].items()
        ):
            raise ValueError("source changed since freeze; use the frozen source or a new run directory")
        {"score": score, "time": time_serial, "scale": scaling}[args.step](args.out, protocol)


if __name__ == "__main__":
    main()
