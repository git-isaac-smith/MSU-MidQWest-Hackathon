"""Paired circuit-level decoder study, with separate calibration, timing and test seeds."""

import argparse
import csv
import hashlib
import json
import platform
from tempfile import TemporaryFile
from shutil import copyfileobj
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import pymatching
import stim
from scipy.stats import binomtest

from .challenge_variants import VARIANT_NAMES, generate_variant_circuit
from .css_routed import CssRoutedDecoder
from .fast_hybrid import FastHybridDecoder
from .stats import wilson95
from .temporal_filter import TemporalFilterDecoder
from .union_find import compile_union_find


METHODS = ("mwpm", "correlated", "union_find", "fast_hybrid", "css_routed", "temporal_lowpass", "local_peel")
RATES = (0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02)
PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)
SOURCES = ("circuit_frontier", "css_routed", "fast_hybrid", "exact_small", "circuits", "temporal_filter",
           "union_find", "challenge_variants")


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def versions():
    return {s: version(s) for s in ("numpy", "stim", "pymatching", "stabilizer-qec", "scipy")}


def source_hashes():
    return {s: hashlib.sha256(Path(__file__).with_name(s + ".py").read_bytes()).hexdigest() for s in SOURCES}


def configuration(smoke):
    cases = []

    def add(study, d, p, basis, variant, shots):
        index = len(cases)
        base = (61900000 if smoke else 87200000) + 10 * index
        cases.append(dict(case=f"{study}-d{d}-{basis}-p{p:g}-{variant}", study=study, distance=d, rounds=d, p=p,
                          basis=basis, variant=variant, seeds=[base, base + 1], shots_per_seed=shots,
                          calibration_seed=50170000 + index, timing_seed=53270000 + index,
                          fast_options=({"correlated_fallback": True, "correlated_min_detectors": round(3.5 + 3500 * p)}
                                        if d == 5 and p <= .005 else {}),
                          timing_batch_size=64 if smoke else 2048, timing_repeats=3 if smoke else 5,
                          latency_shots=8 if smoke else 96))

    add("primary", 3, .005, "z", "ordinary", 512 if smoke else 150000)
    if smoke:
        add("stress", 5, .005, "x", "measurement_burst", 128)
    else:
        for d in (5, 7):
            add("confirmation", d, .005, "z", "ordinary", 75000)
        for p in RATES:
            for d in (3, 5, 7):
                for basis in ("z", "x"):
                    add("sweep", d, p, basis, "ordinary", 10000)
        for d in (3, 5, 7):
            for variant in VARIANT_NAMES[1:]:
                for basis in ("z", "x"):
                    add("stress", d, .005, basis, variant, 10000)
    return dict(methods=METHODS, cases=cases, versions=versions(), sources=source_hashes(),
                noise="Provisional simple_uniform circuit-level depolarization, reset, readout and data idle",
                primary="fast_hybrid vs mwpm at d3 Z p=.005: fewer logical failures, exact paired two-sided "
                        "McNemar p<.05 and lower median interleaved warm batch time.",
                secondary="Prespecified d5 and d7 Z p=.005 css_routed vs mwpm. Report both comparisons, "
                          "using Bonferroni .025 per paired test; timings are descriptive.",
                routing="css_routed: above 94th percentile of detector counts on 8192 separate unlabeled shots",
                lowpass="reflective temporal DCT, 0.20 cycles/round, 0.5 hard threshold; endpoints preserved",
                platform=platform.platform(), cpu=platform.processor(), python=platform.python_version())


def compile_method(name, circuit, model, case):
    if name in ("mwpm", "correlated"):
        correlated = name == "correlated"
        matcher = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
        return partial(matcher.decode_batch, **PACKED, enable_correlations=correlated)
    if name == "fast_hybrid":
        return partial(FastHybridDecoder(model, **case["fast_options"]).decode_batch, **PACKED)
    if name == "union_find":
        return compile_union_find(model).decode_batch
    if name == "css_routed":
        return CssRoutedDecoder(circuit, model, calibration_seed=case["calibration_seed"]).decode_batch
    if name in ("local_peel", "temporal_lowpass"):
        mode = "guarded" if name == "local_peel" else "lowpass"
        return TemporalFilterDecoder(circuit, cutoff=.20, threshold=.5, mode=mode).decode_batch
    raise ValueError(f"unknown method {name}")


def predict_bits(decoder, packed):
    predicted = np.asarray(decoder(packed))
    if predicted.shape != (len(packed), 1) or predicted.dtype.kind not in "bui" or np.any(predicted > 1):
        raise ValueError("Expected one binary observable per shot")
    return predicted[:, 0]


def time_methods(circuit, methods, case):
    sampler = circuit.compile_detector_sampler(seed=case["timing_seed"])
    warm = sampler.sample(192, bit_packed=True)
    n = case["timing_batch_size"]
    batches = [sampler.sample(n, bit_packed=True) for _ in range(case["timing_repeats"])]
    latencies = sampler.sample(case["latency_shots"], bit_packed=True)
    times, single = {name: [] for name in methods}, {}
    rng = np.random.default_rng(case["timing_seed"])
    for fn in methods.values():
        fn(warm)
    for batch in batches:
        for name in rng.permutation(list(methods)):
            start = perf_counter_ns()
            methods[name](batch)
            times[name].append((perf_counter_ns() - start) / n / 1000)
    for name, fn in methods.items():
        expected = predict_bits(fn, latencies)
        single[name] = []
        for i, answer in enumerate(expected):
            shot = latencies[i:i + 1]
            start = perf_counter_ns()
            prediction = fn(shot)
            single[name].append((perf_counter_ns() - start) / 1000)
            if prediction[0, 0] != answer:
                raise ValueError(f"{name} single and batch predictions disagree")
    return times, single


def run_case(folder, case, plan):
    folder.mkdir(exist_ok=True)
    circuit = generate_variant_circuit(case["variant"], **{k: case[k] for k in ("distance", "rounds", "basis", "p")})
    circuit.to_file(folder / "circuit.stim")
    model = circuit.detector_error_model(decompose_errors=True)
    methods, setup = {}, {}
    for name in METHODS:
        start = perf_counter()
        methods[name] = compile_method(name, circuit, model, case)
        setup[name] = perf_counter() - start
    times, single = time_methods(circuit, methods, case)
    counts = {m: np.zeros(3, dtype=np.int64) for m in methods}
    spectral_records = []
    for seed in case["seeds"]:
        detectors, truth = circuit.compile_detector_sampler(seed=seed).sample(
            case["shots_per_seed"], separate_observables=True, bit_packed=True)
        if case["distance"] == 7 and case["basis"] == "z" and case["p"] == .005 and case["study"] in ("sweep", "stress"):
            spectral_records.append(detectors)
        detectors.flags.writeable = False
        archive = dict(detectors=detectors, truth=truth)
        for name, fn in methods.items():
            prediction = np.concatenate([predict_bits(fn, detectors[i:i + 4096]) for i in range(0, len(detectors), 4096)])
            failed = prediction != truth[:, 0]
            if name == "mwpm":
                baseline_failed = failed
            counts[name] += [failed.sum(), (baseline_failed & ~failed).sum(), (~baseline_failed & failed).sum()]
            archive[name] = np.packbits(prediction, bitorder="little")
        np.savez_compressed(folder / f"seed-{seed}.npz", **archive)
    if spectral_records:
        from .spectral_diagnostics import summarize
        summarize(folder, circuit, np.concatenate(spectral_records))
    rows = []
    n = case["shots_per_seed"] * len(case["seeds"])
    for name, (err, fixed, added) in counts.items():
        low, high = wilson95(int(err), n)
        batch = float(np.median(times[name]))
        rows.append({k: case[k] for k in ("case", "basis", "distance", "rounds", "p", "variant", "study")} |
                    dict(method=name, shots=n, errors=int(err), ler_percent=100 * int(err) / n,
                         wilson_low=100 * low, wilson_high=100 * high, batch_us_shot=batch,
                         batch_us_round=batch / case["rounds"], single_median_us=float(np.median(single[name])),
                         single_p95_us=float(np.percentile(single[name], 95)), setup_s=setup[name],
                         fixed=int(fixed), added=int(added),
                         paired_p=float(binomtest(fixed, fixed + added).pvalue) if fixed + added else 1.0))
    save_json(folder / "manifest.json", case | dict(versions=plan["versions"], timings=times, single_us=single,
                                                    setup_s=setup, detector_coordinates=circuit.get_detector_coordinates()))
    save_json(folder / "rows.json", rows)
    return rows


def run(out, smoke):
    out.mkdir(parents=True, exist_ok=True)
    plan = configuration(smoke)
    plan = json.loads(json.dumps(plan))
    protocol = out / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8")) != plan:
            raise ValueError("Choose a new run directory for a different protocol or source revision")
    elif any(out.iterdir()):
        raise ValueError("Choose an empty directory")
    else:
        save_json(protocol, plan)
    base = out / "cases"
    base.mkdir(exist_ok=True)
    rows = []
    for case in plan["cases"]:
        folder = base / case["case"]
        saved = folder / "rows.json"
        block = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else run_case(folder, case, plan)
        print(case["case"], " | ".join(f"{r['method']} {r['errors']}/{r['shots']} {r['batch_us_shot']:.3f}us"
                                      for r in block if r["method"] in ("mwpm","fast_hybrid","css_routed")), flush=True)
        rows.extend(block)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def deploy(case_folder, name, source, output):
    case = json.loads((case_folder / "manifest.json").read_text(encoding="utf-8"))
    circuit = stim.Circuit.from_file(case_folder / "circuit.stim")
    decoder = compile_method(name, circuit, circuit.detector_error_model(decompose_errors=True), case)
    width = (circuit.num_detectors + 7) // 8
    if source.stat().st_size % width:
        raise ValueError("Partial detector row in Stim b8 file")
    if output.exists():
        raise FileExistsError(output)
    pad = (255 << (circuit.num_detectors % 8)) & 255 if circuit.num_detectors % 8 else 0
    with source.open("rb") as incoming, TemporaryFile() as staged:
        while block := incoming.read(4096 * width):
            records = np.frombuffer(block, dtype=np.uint8).reshape(-1, width)
            if pad and np.any(records[:, -1] & pad):
                raise ValueError("Nonzero padding in detector row")
            prediction = predict_bits(decoder, records)
            staged.write(prediction.astype(np.uint8).tobytes())
        staged.seek(0)
        with output.open("xb") as outgoing:
            copyfileobj(staged, outgoing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    study = sub.add_parser("run")
    study.add_argument("--out", type=Path, required=True)
    study.add_argument("--smoke", action="store_true")
    pred = sub.add_parser("predict")
    pred.add_argument("--case", type=Path, required=True)
    pred.add_argument("--method", choices=METHODS, required=True)
    pred.add_argument("--detectors", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out, args.smoke)
    else:
        deploy(args.case, args.method, args.detectors, args.output)


if __name__ == "__main__":
    main()
