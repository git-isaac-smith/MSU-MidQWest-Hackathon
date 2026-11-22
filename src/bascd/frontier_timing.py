"""Repeat the three prespecified speed comparisons on fresh batches."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import stim

from .circuit_frontier import compile_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new timing output file")
    result = []
    for d in (3, 5, 7):
        study = "primary" if d == 3 else "confirmation"
        folder = args.run / "cases" / f"{study}-d{d}-z-p0.005-ordinary"
        case = json.loads((folder / "manifest.json").read_text())
        circuit = stim.Circuit.from_file(folder / "circuit.stim")
        model = circuit.detector_error_model(decompose_errors=True)
        method = "fast_hybrid" if d == 3 else "css_routed"
        methods = {name: compile_method(name, circuit, model, case) for name in ("mwpm", method)}
        seed = 89600000 + d
        rng = np.random.default_rng(seed)
        sampler = circuit.compile_detector_sampler(seed=seed)
        warmup = sampler.sample(4096, bit_packed=True)
        for fn in methods.values():
            fn(warmup)
        n = 8192
        runs = {name: [] for name in methods}
        for _ in range(21):
            batch = sampler.sample(n, bit_packed=True)
            for name in rng.permutation(list(methods)):
                start = perf_counter_ns()
                methods[name](batch)
                runs[name].append((perf_counter_ns() - start) / n / 1000)
        ratios = np.array(runs["mwpm"]) / runs[method]
        interval = np.quantile(np.median(rng.choice(ratios, (20000, len(ratios))), axis=1), [.025, .975])
        result.append(dict(distance=d, candidate=method, seed=seed, shots_per_batch=n, repetitions=len(ratios),
                           faster_batches=int((ratios > 1).sum()), median_paired_speedup=float(np.median(ratios)),
                           paired_batch_bootstrap95=interval.tolist(), times_us_shot=runs,
                           median_us_shot={k: float(np.median(v)) for k, v in runs.items()}))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(dict(protocol="21 fresh paired batches, randomized decoder order, separate warm-up, no sampling "
                                "or construction in timer; interval describes run-to-run variation on this host.",
                       platform=platform.platform(), rows=result), stream, indent=2)


if __name__ == "__main__":
    main()
