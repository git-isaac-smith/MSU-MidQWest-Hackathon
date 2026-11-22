"""Predeclared paired higher-noise stress benchmark for six fixed-circuit decoders."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

import numpy as np
import psutil
from scipy.stats import binomtest
import stim

from bascd.bath_prior import BathPrior, bath_prior_dem, load_bath_prior
from bascd.circuits import memory_circuit
from bascd.css_routed import CssRoutedDecoder
from bascd.decoders import DecoderSpec, compile_decoder
from bascd.stats import wilson95
from bascd.union_find import compile_union_find

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "experiments/iteration-03/bath-calibration/bath-profile.json"
DISTANCES = (7, 15, 31)
P = .01
BATCH = 128
SHOTS = 2048
SEED_ROOT = 69852000
METHODS = ("mwpm", "correlated", "expanded_corr", "clock_corr", "union_find", "css_routed")
LIMIT_BYTES = int(1.5 * 1024**3)
LIMIT_SECONDS = 240


def digest(*items):
    hash_value = hashlib.sha256()
    for item in items:
        hash_value.update(item)
    return hash_value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def store(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = [dict(distance=d, basis="z", rounds=d, seed=SEED_ROOT + d, calibration_seed=SEED_ROOT + 2000 + d)
             for d in DISTANCES]
    protocol = dict(p=P, model="provisional simple_uniform", cases=cases, shots=SHOTS, batch_shots=BATCH,
                    methods=METHODS, css_calibration_shots=1024, css_fraction=.06,
                    profile_sha256=digest(PROFILE.read_bytes()), method_rss_limit_bytes=LIMIT_BYTES,
                    method_timeout_seconds=LIMIT_SECONDS,
                    versions={pkg: version(pkg) for pkg in ("numpy", "stim", "pymatching", "stabilizer-qec", "scipy")},
                    timing="One worker per method, shared host; setup, sample generation, scoring excluded. "
                           "Median of distinct warm full-sized batch calls, with required unpacking included.")
    store(out / "protocol.json", protocol)
    for case in cases:
        d = case["distance"]
        folder = out / f"d{d}-z"
        folder.mkdir()
        circuit = memory_circuit("z", d, d, P)
        circuit.to_file(folder / "circuit.stim")
        detectors, truth = circuit.compile_detector_sampler(seed=case["seed"]).sample(
            SHOTS, separate_observables=True, bit_packed=True
        )
        np.savez_compressed(folder / "records.npz", detectors=detectors, truth=truth)
        store(folder / "case.json", dict(**case, detectors=circuit.num_detectors,
                                         circuit_sha256=digest(str(circuit).encode()),
                                         sample_sha256=digest(detectors, truth)))


def make_decoder(circuit, case, name):
    if name in ("expanded_corr", "clock_corr"):
        expand = name == "expanded_corr"
        prior = BathPrior("clock", (1/3,) * 3) if expand else load_bath_prior(PROFILE, channel="clock")
        model = bath_prior_dem(circuit, basis="z", distance=case["distance"], rounds=case["rounds"],
                               p=P, prior=prior, expand_isotropic=expand)
    else:
        model = circuit.detector_error_model(decompose_errors=True)
    if name == "union_find":
        return compile_union_find(model).decode_batch
    if name == "css_routed":
        return CssRoutedDecoder(circuit, model, calibration_seed=case["calibration_seed"],
                                fraction=.06, calibration_shots=1024).decode_batch
    spec = DecoderSpec("mwpm" if name == "mwpm" else "correlated_mwpm")
    decoder = compile_decoder(model, spec)

    def decode(packed):
        detectors = np.unpackbits(packed, count=circuit.num_detectors, axis=1, bitorder="little")
        return decoder.decode_batch(detectors)
    return decode


def worker(folder, name):
    case = load(folder / "case.json")
    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    with np.load(folder / "records.npz", allow_pickle=False) as records:
        packed = records["detectors"]
    decode = make_decoder(circuit, case, name)
    warm = circuit.compile_detector_sampler(seed=case["calibration_seed"] + 1000).sample(8, bit_packed=True)
    decode(warm)
    samples = []
    parts = []
    for offset in range(0, len(packed), BATCH):
        records = packed[offset:offset + BATCH]
        start = time.perf_counter_ns()
        values = decode(records)
        samples.append((time.perf_counter_ns() - start) / (1000 * len(records)))
        if values.shape != (len(records), 1) or np.any((values != 0) & (values != 1)):
            raise ValueError(f"Unexpected output for {name}")
        parts.append(values[:, 0].astype(np.uint8))
    prediction = np.concatenate(parts)
    np.save(folder / f"{name}.npy", prediction, allow_pickle=False)
    store(folder / f"{name}.json", dict(status="ok", prediction_sha256=digest(prediction),
                                         batch_times_us_per_shot=samples, batch_us_per_shot=median(samples)))


def supervise(folder, name):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    child = subprocess.Popen([sys.executable, __file__, "worker", "--out", str(folder), "--method", name], env=env)
    process = psutil.Process(child.pid)
    start = time.monotonic()
    reason = ""
    while child.poll() is None:
        try:
            members = [process, *process.children(recursive=True)]
            if sum(member.memory_info().rss for member in members if member.is_running()) > LIMIT_BYTES:
                reason = "Worker exceeded 1.5 GiB RSS"
        except psutil.NoSuchProcess:
            break
        if time.monotonic() - start > LIMIT_SECONDS:
            reason = "Worker exceeded 240 seconds"
        if reason:
            for member in reversed(members):
                try:
                    member.kill()
                except psutil.NoSuchProcess:
                    pass
            break
        time.sleep(.05)
    code = child.wait()
    marker = folder / f"{name}.json"
    if not marker.exists():
        store(marker, dict(status="resource_limit" if reason else "error", reason=reason or f"Exit code {code}"))
    print(f"{folder.name} {name}: {load(marker)['status']}", flush=True)


def compare(candidate, reference, label):
    wins = int((reference & ~candidate).sum())
    losses = int((~reference & candidate).sum())
    n, discordant = len(candidate), wins + losses
    freq = binomtest(discordant, n).proportion_ci(confidence_level=.975)
    direction = binomtest(losses, discordant).proportion_ci(confidence_level=.975) if discordant else (0, 1)
    limits = [100 * f * (2 * p - 1) for f in freq for p in direction]
    return {f"{label}_{key}": value for key, value in dict(
        recovered=wins, regressed=losses, delta_pp=100 * (losses - wins) / n,
        paired95_low_pp=min(limits), paired95_high_pp=max(limits),
        mcnemar_p=binomtest(losses, discordant).pvalue if discordant else 1.0,
    ).items()}


def collect(out, *, replay):
    protocol = load(out / "protocol.json")
    if protocol["profile_sha256"] != digest(PROFILE.read_bytes()):
        raise ValueError("Bath profile hash differs from the recorded protocol")
    if protocol["versions"]["stim"] != stim.__version__:
        raise ValueError("Stim version differs from the recorded protocol")
    rows = []
    for case in protocol["cases"]:
        folder = out / f'd{case["distance"]}-z'
        meta = load(folder / "case.json")
        circuit = stim.Circuit.from_file(folder / "circuit.stim")
        with np.load(folder / "records.npz", allow_pickle=False) as data:
            packed, truth = data["detectors"], data["truth"]
        assert packed.dtype == truth.dtype == np.uint8
        assert packed.shape == (protocol["shots"], (circuit.num_detectors + 7) // 8)
        assert truth.shape == (protocol["shots"], 1) and np.all(truth <= 1)
        assert digest(str(circuit).encode()) == meta["circuit_sha256"]
        assert digest(packed, truth) == meta["sample_sha256"]
        if replay:
            expected_circuit = memory_circuit(case["basis"], case["distance"], case["rounds"], protocol["p"])
            assert circuit == expected_circuit
            regenerated = circuit.compile_detector_sampler(seed=case["seed"]).sample(
                len(packed), separate_observables=True, bit_packed=True
            )
            np.testing.assert_array_equal(packed, regenerated[0])
            np.testing.assert_array_equal(truth, regenerated[1])
        masks, entries = {}, []
        for name in METHODS:
            status = load(folder / f"{name}.json")
            result = dict(distance=case["distance"], data_qubits=case["distance"]**2, method=name,
                          status=status["status"], reason=status.get("reason", ""))
            if status["status"] == "ok":
                prediction = np.load(folder / f"{name}.npy", allow_pickle=False)
                assert prediction.shape == (len(packed),) and digest(prediction) == status["prediction_sha256"]
                bad = prediction != truth[:, 0]
                masks[name] = bad
                count = int(bad.sum())
                lo, hi = wilson95(count, len(bad))
                result.update(shots=len(bad), errors=count, ler_pct=100*count/len(bad), wilson_low_pct=100*lo,
                              wilson_high_pct=100*hi, batch_us_per_shot=status["batch_us_per_shot"])
            entries.append(result)
        for result in entries:
            name = result["method"]
            if name in masks:
                for ref in ("mwpm", "expanded_corr"):
                    if ref in masks:
                        result.update(compare(masks[name], masks[ref], ref))
            rows.append(result)
    metric = out / "metrics.csv"
    if replay:
        with metric.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(rows)
        for row, expected in zip(saved, rows, strict=True):
            assert row == {name: str(expected.get(name, "")) for name in row}
        print(f"Recreated and scored all {len(rows)} rows.")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with metric.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify", "worker"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.out, args.method)
    elif args.action == "verify":
        collect(args.out, replay=True)
    else:
        prepare(args.out)
        for distance in DISTANCES:
            for name in METHODS:
                supervise(args.out / f"d{distance}-z", name)
        collect(args.out, replay=False)


if __name__ == "__main__":
    main()



