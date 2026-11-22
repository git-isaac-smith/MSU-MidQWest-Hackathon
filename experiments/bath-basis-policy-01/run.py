"""Confirm a fixed memory-basis policy on common, unmodified Stim shots."""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from bascd.bath_prior import BathPrior, bath_prior_dem, bath_prior_from_profile
from bascd.circuits import memory_circuit
from bascd.decoders import DecoderSpec, compile_decoder
from bascd.measure import measure_decoder, sample_circuit
from bascd.records import manifest, save_csv, save_json, scenarios
from bascd.stats import paired_counts, paired_ler_difference


ROOT_SEED = 562491030
SHOTS = 125_000
SPLIT = "final_test"
PROFILE = Path("experiments/iteration-03/bath-calibration/bath-profile.json")
POLICY = {"x": "bath", "z": "equal"}
REFERENCES = ("equal", "bath")
SPEC = DecoderSpec("correlated_mwpm")


def sample_case(case, seed):
    params = dict(**asdict(case), rounds=case.distance)
    physical = memory_circuit(**params)
    data = sample_circuit(lambda: physical, seed, SPLIT, SHOTS,
                          dict(**params, noise_model="simple_uniform", noise_status="provisional"))
    return physical, params, data


def measure_case(case, index, priors):
    physical, params, data = sample_case(case, case.seed(ROOT_SEED, SPLIT))
    row = {key: value for key, value in data.metadata.items()}
    row["policy_prior"] = POLICY[case.basis]
    masks = {}
    for name in REFERENCES[::(-1 if index % 2 else 1)]:
        model = bath_prior_dem(physical, **params, prior=priors[name], expand_isotropic=name == "equal")
        row[f"{name}_dem_sha256"] = hashlib.sha256(str(model).encode()).hexdigest()
        measured, masks[name] = measure_decoder(
            data, lambda: compile_decoder(model, SPEC), SPEC.decoder_type, SPEC.to_dict(), repeats=1
        )
        row[f"{name}_failures"] = measured["failures"]
        row[f"{name}_batch_s"] = measured["median_batch_s"]
    candidate = row["policy_prior"]
    row["policy_failures"] = row[f"{candidate}_failures"]
    row["policy_batch_s"] = row[f"{candidate}_batch_s"]
    for name in REFERENCES:
        counts = paired_counts(masks[name], masks[candidate])
        row[f"{name}_recovered"] = counts["mwpm_only_failed"]
        row[f"{name}_introduced"] = counts["correlated_only_failed"]
    return row


def summarize(rows):
    shots = SHOTS * len(rows)
    methods = {}
    for name in ("policy", *REFERENCES):
        failures = sum(int(row[f"{name}_failures"]) for row in rows)
        seconds = sum(float(row[f"{name}_batch_s"]) for row in rows)
        methods[name] = dict(failures=failures, ler=failures / shots, batch_us_per_shot=seconds * 1e6 / shots)
    pairs = {}
    for name in REFERENCES:
        pairs[name] = paired_ler_difference(
            (SHOTS for _ in rows),
            (int(row[f"{name}_recovered"]) for row in rows),
            (int(row[f"{name}_introduced"]) for row in rows),
        )
    return dict(shots=shots, methods=methods, policy_vs=pairs)


def summaries(rows):
    return dict(pooled=summarize(rows),
                by_basis={basis: summarize([row for row in rows if row["basis"] == basis]) for basis in POLICY})


def run(out):
    raw_profile = PROFILE.read_bytes()
    profile = json.loads(raw_profile.decode("utf-8-sig"))
    priors = dict(equal=BathPrior("clock", (1 / 3, 1 / 3, 1 / 3)), bath=bath_prior_from_profile(profile))
    cases = scenarios()
    protocol = manifest(["python", *sys.argv], ROOT_SEED, SHOTS, 1)
    protocol.update(
        scenarios=[dict(**asdict(case), seed=case.seed(ROOT_SEED, SPLIT)) for case in cases], split=SPLIT,
        frozen_basis_policy=POLICY, primary="policy vs always equal", secondary="policy vs always bath",
        profile=str(PROFILE), profile_sha256=hashlib.sha256(raw_profile).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration_source=profile.get("calibration_source"), bath_clock_time=profile["clock"]["time"],
        prior_ratios={name: prior.ratios for name, prior in priors.items()},
        timing="One warm full batch per hypothesis and scenario; alternate hypothesis order. "
               "Basis policy selects its graph at setup, so its timing uses that hypothesis.",
        interval="Candidate minus reference, approximate paired 95% interval stratified by scenario.",
    )
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "manifest.json", protocol)
    rows = []
    for index, case in enumerate(cases):
        rows.append(measure_case(case, index, priors))
        print(f"{index + 1}/{len(cases)} scenarios complete", flush=True)
    save_csv(out / "scenarios.csv", rows)
    save_json(out / "summary.json", summaries(rows))


def verify(out, replay):
    protocol = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    with (out / "scenarios.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if protocol["root_seed"] != ROOT_SEED or protocol["frozen_basis_policy"] != POLICY:
        raise ValueError("run does not match frozen policy and root seed")
    if protocol["scenarios"] != [dict(**asdict(case), seed=case.seed(ROOT_SEED, SPLIT)) for case in scenarios()]:
        raise ValueError("unexpected scenario seed plan")
    if len({row["sample_sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample fingerprint")
    for index, (case, row) in enumerate(zip(scenarios(), rows, strict=True)):
        if (int(row["distance"]), float(row["p"]), row["basis"]) != (case.distance, case.p, case.basis):
            raise ValueError(f"unexpected scenario {row}")
        if int(row["seed"]) != case.seed(ROOT_SEED, SPLIT) or int(row["shots"]) != SHOTS:
            raise ValueError(f"unexpected seed or shot count for {case}")
        selected = POLICY[case.basis]
        if row["policy_prior"] != selected or row["policy_failures"] != row[f"{selected}_failures"]:
            raise ValueError(f"policy disagrees with its frozen hypothesis for {case}")
        for name in REFERENCES:
            recovered = int(row[f"{name}_recovered"])
            introduced = int(row[f"{name}_introduced"])
            reference = int(row[f"{name}_failures"])
            candidate = int(row["policy_failures"])
            if not 0 <= recovered <= reference <= SHOTS or not 0 <= introduced <= SHOTS - reference:
                raise ValueError(f"invalid paired count for {case} vs {name}")
            if reference - recovered + introduced != candidate or name == selected and (recovered or introduced):
                raise ValueError(f"paired counts do not reconcile for {case} vs {name}")
        if replay:
            _, _, data = sample_case(case, int(row["seed"]))
            for key in ("circuit_sha256", "sample_sha256"):
                if row[key] != data.metadata[key]:
                    raise ValueError(f"{key} mismatch for {case}")
    if summaries(rows) != json.loads((out / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("summary does not reconcile")
    print(f"Verified {len(rows)} scenarios and {SHOTS * len(rows):,} paired shots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replay", action="store_true", help="regenerate sample fingerprints during verification")
    args = parser.parse_args()
    if args.verify or args.replay:
        verify(args.out, args.replay)
    else:
        run(args.out)
