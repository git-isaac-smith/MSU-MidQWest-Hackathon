"""Compare bath-initialized matching on identical fixed-noise surface-code shots."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .bath_prior import BathPrior, bath_prior_dem, bath_prior_from_profile
from .circuits import memory_circuit
from .decoders import CHOICES, compile_decoder
from .measure import measure_decoder, sample_circuit
from .records import manifest, save_csv, save_json, scenarios
from .stats import paired_counts, paired_ler_difference


def compare_case(case, priors, seed, shots, repeats, latency_shots, index):
    params = dict(**asdict(case), rounds=case.distance)
    physical = memory_circuit(**params)
    models, setup = {}, {}
    for channel, prior in priors.items():
        start = perf_counter()
        models[channel] = bath_prior_dem(physical, **params, prior=prior,
                                         expand_isotropic=channel == "expanded_uniform")
        setup[channel] = perf_counter() - start
    # The sampler and evaluator use only the original physical circuit.
    data = sample_circuit(lambda: physical, case.seed(seed, "sweep"), "sweep", shots,
                          dict(**params, noise_model="simple_uniform", noise_status="provisional"))
    models["native"] = data.model
    # The native DEM is already included in simulation_setup_s. Measure standalone
    # conversion separately on the same circuit for comparable offline setup costs.
    start = perf_counter()
    physical.detector_error_model(decompose_errors=True)
    setup["native"] = perf_counter() - start
    digests = {name: hashlib.sha256(str(model).encode()).hexdigest() for name, model in models.items()}
    order = [(channel, spec) for channel in ("native", *priors) for spec in CHOICES[:2]]
    offset = index % len(order)
    rows, masks = [], {}
    for position, (channel, spec) in enumerate(order[offset:] + order[:offset], 1):
        kind = spec.decoder_type
        settings = dict(**spec.to_dict(), prior=channel,
                        idle_fault_xyz_ratio=priors[channel].ratios if channel in priors else [1/3] * 3)
        row, masks[channel, kind] = measure_decoder(
            data, lambda: compile_decoder(models[channel], spec), kind, settings, repeats, latency_shots)
        rows.append(dict(**row, prior=channel, hypothesis_dem_sha256=digests[channel],
                         hypothesis_dem_build_s=setup[channel], timing_position=position))
    paired = []
    for channel in priors:
        for spec in CHOICES[:2]:
            kind = spec.decoder_type
            references = ("native",) if channel == "expanded_uniform" else ("native", "expanded_uniform")
            for ref in references:
                counts = paired_counts(masks[ref, kind], masks[channel, kind])
                paired.append(dict(**data.metadata, decoder=kind, prior=channel, reference=ref,
                                   both_correct=counts["both_correct"], both_failed=counts["both_failed"],
                                   recovered=counts["mwpm_only_failed"], lost=counts["correlated_only_failed"]))
    return rows, paired


def summarize(rows, paired):
    summaries = []
    for channel, kind in sorted({(r["prior"], r["decoder"]) for r in rows}):
        group = [r for r in rows if (r["prior"], r["decoder"]) == (channel, kind)]
        shots = sum(r["shots"] for r in group)
        failures = sum(r["failures"] for r in group)
        point = dict(prior=channel, decoder=kind, shots=shots, failures=failures, ler=failures/shots,
                     batch_us_per_shot=1e6 * sum(r["median_batch_s"] for r in group) / shots)
        pairs = [r for r in paired if (r["prior"], r["decoder"]) == (channel, kind)]
        for ref in sorted({r["reference"] for r in pairs}):
            group_pairs = [r for r in pairs if r["reference"] == ref]
            point[f"paired_vs_{ref}"] = paired_ler_difference(
                (r["shots"] for r in group_pairs), (r["recovered"] for r in group_pairs),
                (r["lost"] for r in group_pairs))
        summaries.append(point)
    return summaries


def run(profile_path, out, cases, seed, shots, repeats, latency_shots, command):
    if shots < 2 or repeats < 1 or not 0 <= latency_shots <= shots or not 0 <= seed < 2**64:
        raise ValueError("invalid seed, shot count, latency sample size, or timing repeats")
    raw_profile = profile_path.read_bytes()
    profile = json.loads(raw_profile.decode("utf-8-sig"))
    bath_priors = {channel: bath_prior_from_profile(profile, channel=channel) for channel in ("clock", "selected")}
    priors = {"expanded_uniform": BathPrior("clock", (1/3, 1/3, 1/3)), **bath_priors}
    cases = list(cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("the experiment needs at least one scenario and each scenario must be unique")
    protocol = manifest(command, seed, shots, repeats)
    protocol.update(
        scenarios=[asdict(case) for case in cases], split="sweep", latency_shots=latency_shots,
        profile=str(profile_path), profile_sha256=hashlib.sha256(raw_profile).hexdigest(),
        calibration_source=profile.get("calibration_source"), primary_contrast="clock vs native within matcher",
        secondary_contrast="selected vs native within matcher, using an offline alternative prior",
        representation_control="expanded_uniform uses three independent Pauli errors encoding the original "
                               "uniform distribution. Compare each bath profile with both native and expanded_uniform.",
        decoder_order="rotate the eight configurations by one position per scenario",
        bath_priors={channel: dict(time=profile[channel]["time"], xyz=prior.ratios)
                     for channel, prior in bath_priors.items()},
        mapping="Only the relative X/Y/Z mixture of pre-round data errors enters the matching model. "
                "Every matcher receives the same fixed-clock detector samples. All shots count.",
        interval="Unadjusted paired, stratified normal approximation for the difference in LER, "
                 "omitted below 30 discordant shots. Per-scenario LER intervals use Wilson 95%.",
    )
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "manifest.json", protocol)
    rows, pairs = [], []
    for index, case in enumerate(cases):
        measurements, comparisons = compare_case(case, priors, seed, shots, repeats, latency_shots, index)
        rows.extend(measurements)
        pairs.extend(comparisons)
    summary = summarize(rows, pairs)
    save_csv(out / "benchmark.csv", rows)
    save_csv(out / "paired.csv", pairs)
    save_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="bath-profile.json from bath_initialize")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=602711)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-shots", type=int, default=64)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[.001, .003, .005, .01])
    parser.add_argument("--bases", choices=("x", "z"), nargs="+", default=["x", "z"])
    args = parser.parse_args()
    try:
        result = run(args.profile, args.out, scenarios(args.distances, args.probabilities, args.bases),
                     args.seed, args.shots, args.repeats, args.latency_shots,
                     ["python", "-m", "bascd.bath_transfer", *sys.argv[1:]])
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
