"""Blind-MWPM stress test under a latent spatial Poisson burst process."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching

from .burst_samples import BurstSampler, poisson_parity_probability
from .circuits import memory_circuit
from .stats import wilson95

EXPOSURES = ("poisson", "one", "independent")
PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


def event_draws(rng, n, geometry, density, exposure):
    """A shared realization of burst centers and uniform flip draws for all xi."""
    if exposure == "one":
        counts = np.ones(n, dtype=int)
    elif exposure == "poisson":
        counts = rng.poisson(density * geometry.n_sites * geometry.rounds, n)
    else:
        raise ValueError("expected poisson or one")
    ids = np.repeat(np.arange(n), counts)
    rounds = rng.integers(geometry.rounds, size=len(ids))
    centers = rng.integers(geometry.n_sites, size=len(ids))
    uniforms = rng.random((len(ids), geometry.n_sites))
    return ids, rounds, centers, uniforms, counts


def apply_events(draws, kernel, rounds):
    ids, times, centers, uniforms, counts = draws
    flips = np.zeros((len(counts), rounds, kernel.shape[1]), dtype=bool)
    # Two X flips on one site in one round cancel, including overlapping bursts.
    np.logical_xor.at(flips, (ids, times), uniforms < kernel[centers])
    return flips


def save_csv(path, rows):
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    folder,
    *,
    distances=(3, 5, 7),
    lengths=(1.0, 2.0, 4.0),
    density=0.001,
    p=0.005,
    seeds=4,
    shots=25000,
    batch=1024,
    root_seed=93710000,
):
    if (
        not distances
        or any(d < 3 or d % 2 == 0 for d in distances)
        or len(set(distances)) != len(distances)
        or not lengths
        or len(set(lengths)) != len(lengths)
        or any(not np.isfinite(x) or x <= 0 for x in lengths)
        or not np.isfinite(density)
        or density < 0
        or not 0 <= p <= 1
        or min(seeds, shots, batch) < 1
        or root_seed < 0
    ):
        raise ValueError(
            "expected distinct odd distances, positive lengths and valid rates, seeds and sizes"
        )
    protocol = dict(
        distances=distances,
        xi=lengths,
        density=density,
        background_p=p,
        basis="Z",
        rounds="distance",
        exposures=("none", *EXPOSURES),
        shots_per_seed=shots,
        seed_count=seeds,
        batch_size=batch,
        root_seed=root_seed,
        seed_scheme="NumPy SeedSequence([root_seed,distance,seed_index]).spawn(4) "
        "in order: Stim background, Poisson, one burst, independent.",
        kernel="0.5 exp(-Euclidean center-to-site distance / xi), in data-qubit spacings",
        density_units="expected burst cores per data qubit per round",
        injection="X flips before first H of extraction round; conditional site draws independent; "
        "uniform on-patch data-site cores and uniform rounds; overlapping X flips XOR",
        controls="no bursts; exactly one core per shot; independent site-round X flips with "
        "same marginal odd-flip probabilities as the Poisson model at each d and xi",
        decoder="ordinary PyMatching compiled only from provisional background simple_uniform DEM; "
        "decoder input is packed detectors only",
        pairing="Within each seed/case all conditions reuse Stim background detectors and observable. "
        "Within an exposure all xi reuse centers and uniform draws. Distances use independent seeds.",
        timing="Inference only, warm decoder, same batch sizes; timing is descriptive.",
        artifacts="seeds-*.npz store failures in little-endian order, keyed by exposure and xi, "
        "and counts of Poisson cores per shot; see replication.csv for seed-index keys.",
        versions={k: version(k) for k in ("stim", "pymatching", "numpy", "scipy")},
        source_sha256={
            file: hashlib.sha256(Path(__file__).with_name(file).read_bytes()).hexdigest()
            for file in ("burst_samples.py", "burst_sweep.py", "circuits.py")
        },
    )
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf8")
    cases = [("none", 0.0)] + [(e, x) for e in EXPOSURES for x in lengths]
    rows, replications, comparisons = [], [], []
    for d in distances:
        circuit = memory_circuit("z", d, d, p)
        circuit.to_file(folder / f"d{d}.stim")
        geometry = BurstSampler(circuit, d)
        assert geometry.n_sites == d * d
        kernels = {x: geometry.kernel(x) for x in lengths}
        # Per site, a Poisson number of independent Bernoulli marks thins to a Poisson
        # count of X flips. An odd count toggles the qubit.
        pi = {x: poisson_parity_probability(density * k.sum(axis=0)) for x, k in kernels.items()}
        matcher = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True)
        )
        matcher.decode_batch(np.zeros((1, (circuit.num_detectors + 7) // 8), np.uint8), **PACKED)
        totals = {k: dict(errors=0, flips=0, cores=0, hit=0, seconds=0.0) for k in cases}
        pairing = {}
        for i in range(seeds):
            children = np.random.SeedSequence([root_seed, d, i]).spawn(4)
            stim_seed = int(children[0].generate_state(1, dtype=np.uint64)[0])
            sampler = circuit.compile_detector_sampler(seed=stim_seed)
            rng = {e: np.random.default_rng(child) for e, child in zip(EXPOSURES, children[1:])}
            per = {k: np.empty(shots, dtype=bool) for k in cases}
            numbers = np.empty(shots, dtype=np.int64)
            for pos in range(0, shots, batch):
                n = min(batch, shots - pos)
                bg, bg_obs = sampler.sample(n, separate_observables=True, bit_packed=True)
                draws = {e: event_draws(rng[e], n, geometry, density, e) for e in ("poisson", "one")}
                numbers[pos : pos + n] = draws["poisson"][-1]
                draws_iid = rng["independent"].random((n, d, geometry.n_sites))
                for key in cases:
                    exposure, xi = key
                    det, truth = bg, bg_obs[:, 0]
                    if exposure != "none":
                        if exposure == "independent":
                            flips = draws_iid < pi[xi]
                        else:
                            flips = apply_events(draws[exposure], kernels[xi], d)
                            counts = draws[exposure][-1]
                            totals[key]["cores"] += int(counts.sum())
                            totals[key]["hit"] += int(np.count_nonzero(counts))
                        totals[key]["flips"] += int(flips.sum())
                        delta, obs_delta = geometry.delta(flips)
                        det, truth = bg ^ delta, bg_obs[:, 0] ^ obs_delta
                    start = perf_counter()
                    pred = matcher.decode_batch(det, **PACKED)[:, 0]
                    totals[key]["seconds"] += perf_counter() - start
                    failed = pred != truth
                    totals[key]["errors"] += int(failed.sum())
                    per[key][pos : pos + n] = failed
            archive = {f"{e}_{x:g}": np.packbits(bits, bitorder="little") for (e, x), bits in per.items()}
            np.savez_compressed(folder / f"seeds-d{d}-{i}.npz", poisson_counts=numbers, **archive)
            for e, x in cases:
                replications.append(
                    dict(
                        distance=d,
                        seed_index=i,
                        stim_seed=stim_seed,
                        exposure=e,
                        xi=x,
                        shots=shots,
                        errors=int(per[e, x].sum()),
                    )
                )
            for xi in lengths:
                for reference in (("none", 0.0), ("independent", xi)):
                    baseline, burst = per[reference], per["poisson", xi]
                    counts = pairing.setdefault((xi, *reference), [0, 0])
                    counts[0] += int((baseline & ~burst).sum())
                    counts[1] += int((~baseline & burst).sum())
        N = seeds * shots
        for (xi, reference, ref_xi), (recovered, added) in pairing.items():
            comparisons.append(
                dict(
                    distance=d,
                    xi=xi,
                    reference=reference,
                    reference_xi=ref_xi,
                    shots=N,
                    burst_only_failed=added,
                    reference_only_failed=recovered,
                    burst_minus_reference_pp=100 * (added - recovered) / N,
                )
            )
        for key, tot in totals.items():
            e, x = key
            low, high = wilson95(tot["errors"], N)
            expected = {"none": 0.0, "one": 1.0, "poisson": density * d**3}.get(e)
            exp_flips = 0.0
            if e in ("poisson", "independent"):
                exp_flips = float(d * pi[x].sum())
            elif e == "one":
                exp_flips = float(kernels[x].sum(axis=1).mean())
            rows.append(
                dict(
                    distance=d,
                    rounds=d,
                    background_p=p,
                    density=density,
                    xi=x,
                    exposure=e,
                    shots=N,
                    errors=tot["errors"],
                    ler_percent=100 * tot["errors"] / N,
                    wilson_low_percent=100 * low,
                    wilson_high_percent=100 * high,
                    mean_bursts=tot["cores"] / N if e != "independent" else None,
                    expected_bursts=expected,
                    any_burst_percent=100 * tot["hit"] / N if e != "independent" else None,
                    mean_burst_flips=tot["flips"] / N,
                    expected_added_site_round_flips=exp_flips,
                    batch_us_shot=1e6 * tot["seconds"] / N,
                )
            )
        print(f"d={d}: {N:,} common backgrounds, {len(cases)} conditions", flush=True)
    for name, records in (("results", rows), ("replication", replications), ("comparisons", comparisons)):
        save_csv(folder / f"{name}.csv", records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--xi", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--density", type=float, default=0.001)
    ap.add_argument("--background-p", type=float, default=0.005)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--shots-per-seed", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=93710000)
    args = ap.parse_args()
    run(
        args.out,
        distances=args.distances,
        lengths=args.xi,
        density=args.density,
        p=args.background_p,
        seeds=args.seeds,
        shots=args.shots_per_seed,
        batch=args.batch,
        root_seed=args.seed,
    )


if __name__ == "__main__":
    main()
