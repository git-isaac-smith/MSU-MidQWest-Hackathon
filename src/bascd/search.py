"""Budgeted offline policy search with sealed validation until proposals close."""

import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock
from jsonschema import ValidationError

from .decoders import CHOICES, Policy, validator
from .measure import Scenario, environment, measure, sample
from .records import manifest, save_csv, save_json, scenarios


def parse_proposal(raw):
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("proposal includes a nonfinite number")
        return number

    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("proposal contains duplicate keys")
        return obj

    return json.loads(raw, parse_float=finite, parse_constant=finite, object_pairs_hook=unique)


def random_policies(seed, budget):
    space = list(itertools.product(CHOICES, repeat=3))
    return [Policy(specs) for specs in random.Random(seed).sample(space, budget)]


def summarize(rows):
    shots = sum(r["shots"] for r in rows)
    failures = sum(r["failures"] for r in rows)
    rate = failures / shots
    # Independent bounded losses, with potentially different rates across the fixed grid.
    radius = math.sqrt(math.log(40) / (2 * shots))
    return dict(shots=shots, failures=failures, ler=rate, ler95_low=max(0.0, rate - radius),
                ler95_high=min(1.0, rate + radius),
                batch_us_per_shot=1e6 * sum(r["median_batch_s"] for r in rows) / shots)


def pareto(trials):
    values = {t["trial"]: (t["validation"]["ler"], t["validation"]["batch_us_per_shot"])
              for t in trials if t["status"] == "ok"}
    return [i for i, (x, y) in values.items() if not any(
        a <= x and b <= y and (a < x or b < y) for a, b in values.values())]


def new_campaign(out, strategy, seed, shots, repeats, budget, search_seed, command):
    if not 1 <= budget <= len(CHOICES)**3:
        raise ValueError(f"budget must be in [1, {len(CHOICES)**3}]")
    grid = scenarios()
    protocol = manifest(command, seed, shots, repeats)
    protocol.update(strategy=strategy, candidate_budget=budget, search_seed=search_seed,
                    scenarios=[asdict(c) for c in grid],
                    splits={s: [c.seed(seed, s) for c in grid] for s in ("train", "validation", "final_test")},
                    selection="validation LER and median batch microseconds per shot",
                    uncertainty="per-scenario Wilson 95%; fixed-grid aggregate Hoeffding 95%",
                    validation_release="after the proposal budget closes", final_test_sampled=False)
    out.mkdir(parents=True, exist_ok=False)
    (out / "measurements").mkdir()
    save_json(out / "manifest.json", protocol)
    save_json(out / "history.json", [])
    export_history(out, protocol, [])


def export_history(out, protocol, trials):
    frontier = pareto(trials)
    history_rows, feedback_trials = [], []
    for trial in trials:
        row = {k: trial[k] for k in ("trial", "status", "policy_id", "proposal", "error")}
        row["pareto"] = trial["trial"] in frontier
        for split in ("train", "validation"):
            for m in ("shots", "failures", "ler", "ler95_low", "ler95_high", "batch_us_per_shot"):
                row[f"{split}_{m}"] = (trial[split] or {}).get(m)
        history_rows.append(row)
        feedback_trials.append({k: trial[k] for k in ("trial", "status", "policy_id", "proposal", "error", "train")})
    fields = ["trial", "status", "policy_id", "proposal", "error", "pareto"] + [
        f"{s}_{m}" for s in ("train", "validation") for m in
        ("shots", "failures", "ler", "ler95_low", "ler95_high", "batch_us_per_shot")]
    save_csv(out / "history.csv", history_rows, fields)
    save_json(out / "pareto.json", frontier)
    feedback = {"version": 1, "strategy": protocol["strategy"], "budget": protocol["candidate_budget"],
                "used": len(trials), "trials": feedback_trials}
    validator("feedback").validate(feedback)
    save_json(out / "feedback.json", feedback)


def get_sample(samples, case, split, protocol):
    key = (case, split)
    if key not in samples:
        samples[key] = sample(case, protocol["root_seed"], split, protocol["shots_per_scenario"])
    return samples[key]


def evaluate(out, trial, split, protocol, samples, references=None):
    policy = Policy.from_dict(trial["proposal"])
    rows = []
    for item in protocol["scenarios"]:
        case = Scenario(**item)
        data = get_sample(samples, case, split, protocol)
        row, failed = measure(data, policy.at(case.distance), protocol["timing_repeats"])
        if references is not None:
            for spec in CHOICES[:2]:
                reference_failed = references[case, spec.decoder_type]
                row[f"{spec.decoder_type}_only_failed"] = int((reference_failed & ~failed).sum())
                row[f"candidate_only_failed_vs_{spec.decoder_type}"] = int((failed & ~reference_failed).sum())
        rows.append(row)
    save_json(out / "measurements" / f"{trial['trial']:03}-{split}.json", rows)
    return summarize(rows)


def finish_validation(out, protocol, trials, samples):
    pending = [t for t in trials if t["status"] == "trained"]
    if not pending:
        return
    rows, references = [], {}
    for item in protocol["scenarios"]:
        case = Scenario(**item)
        data = get_sample(samples, case, "validation", protocol)
        for spec in CHOICES[:2]:
            row, references[case, spec.decoder_type] = measure(data, spec, protocol["timing_repeats"])
            rows.append(row)
    path = out / "validation-references.json"
    if not path.exists():
        save_json(path, rows)
    for trial in pending:
        try:
            trial["validation"] = evaluate(out, trial, "validation", protocol, samples, references)
            trial["status"] = "ok"
        except Exception as error:
            trial.update(status="failed", error=f"validation: {error}")
        save_json(out / "history.json", trials)


def advance(out: Path, proposal_path=None):
    """Run remaining random trials, submit one proposal, or recover interrupted exports."""
    if not (out / "manifest.json").is_file():
        raise ValueError("campaign manifest missing")
    with FileLock(str(out / ".bascd.lock"), timeout=1):
        protocol = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if protocol.get("strategy") not in ("random", "guided"):
            raise ValueError("expected a search campaign")
        if proposal_path is not None and protocol["strategy"] != "guided":
            raise ValueError("external proposals require a guided campaign")
        if protocol["environment"] != environment():
            raise ValueError("campaign software or host differs from the recorded protocol")
        trials = json.loads((out / "history.json").read_text(encoding="utf-8"))
        budget = protocol["candidate_budget"]
        for trial in trials:
            if trial["status"] == "running":
                trial.update(status="failed", error="training interrupted after budget reservation")
        save_json(out / "history.json", trials)
        if proposal_path is not None and len(trials) >= budget:
            raise ValueError("candidate budget exhausted")
        proposals = []
        if protocol["strategy"] == "random":
            proposals = [(p.to_dict(), None) for p in random_policies(protocol["search_seed"], budget)[len(trials):]]
        elif proposal_path is not None:
            raw = proposal_path.read_text(encoding="utf-8-sig")
            try:
                proposals = [(parse_proposal(raw), None)]
            except ValueError as error:
                proposals = [(raw, str(error))]

        samples = {}
        for proposal, parse_error in proposals:
            trial = dict(trial=len(trials) + 1, proposal=proposal, status="running", error=None,
                         policy_id=None, train=None, validation=None)
            trials.append(trial)
            save_json(out / "history.json", trials)
            try:
                if parse_error:
                    raise ValueError(parse_error)
                policy = Policy.from_dict(proposal)
                trial["policy_id"] = policy.policy_id
                if any(t["policy_id"] == policy.policy_id for t in trials[:-1]):
                    raise ValueError("duplicate policy")
            except (ValueError, ValidationError) as error:
                trial.update(status="invalid", error=getattr(error, "message", str(error)))
            else:
                try:
                    trial["train"] = evaluate(out, trial, "train", protocol, samples)
                    trial["status"] = "trained"
                except Exception as error:
                    trial.update(status="failed", error=f"training: {error}")
            save_json(out / "history.json", trials)
            export_history(out, protocol, trials)
            print(f"Trial {trial['trial']}/{budget}: {trial['status']}", flush=True)

        if len(trials) == budget:
            samples.clear()
            finish_validation(out, protocol, trials, samples)
        export_history(out, protocol, trials)
        return trials
