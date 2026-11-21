"""CLI for the Iteration 02 experiments. The original bascd CLI stays available."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from filelock import Timeout
from jsonschema import ValidationError

from .decoders import DISTANCES, Policy
from .measure import PROBABILITIES
from .records import scenarios
from .search import advance, new_campaign, parse_proposal
from .sweep import sweep


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    bench = sub.add_parser("sweep", help="compare ordinary and correlated matching on common shots")
    search = sub.add_parser("search", help="start a fresh random or guided campaign")
    for cmd in (bench, search):
        cmd.add_argument("--out", type=Path, required=True, help="new experiment directory")
        cmd.add_argument("--shots", type=int, default=10000, help="shots per scenario per split")
        cmd.add_argument("--seed", type=int, default=2026, help="root sample seed")
        cmd.add_argument("--repeats", type=int, default=3, help="timed batch repetitions")
    bench.add_argument("--distances", type=int, nargs="+", choices=DISTANCES, default=DISTANCES)
    bench.add_argument("--probabilities", type=float, nargs="+", choices=PROBABILITIES, default=PROBABILITIES)
    bench.add_argument("--bases", nargs="+", choices=("x", "z"), default=("x", "z"))
    bench.add_argument("--latency-shots", type=int, default=128, help="individual calls per decoder and scenario")
    search.add_argument("--strategy", choices=("random", "guided"), default="random")
    search.add_argument("--budget", type=int, default=8)
    search.add_argument("--search-seed", type=int, default=17)
    submit = sub.add_parser("submit", help="charge one proposed policy to a guided campaign")
    resume = sub.add_parser("resume", help="finish interrupted work or rebuild campaign exports")
    for cmd in (submit, resume):
        cmd.add_argument("--out", required=True, type=Path)
    submit.add_argument("proposal", type=Path)
    validate = sub.add_parser("validate", help="validate proposal JSON without running inference")
    validate.add_argument("proposal", type=Path)
    opts = parser.parse_args(args)
    command = subprocess.list2cmdline(["python", "-m", "bascd.experiment", *args])
    try:
        if opts.action in ("sweep", "search"):
            if opts.shots < 1 or opts.repeats < 1 or not 0 <= opts.seed < 2**64:
                raise ValueError("shots and repeats must be positive and seed must be unsigned 64-bit")
        if opts.action == "sweep":
            if not 1 <= opts.latency_shots <= opts.shots:
                raise ValueError("latency shots must be between 1 and shots per scenario")
            cases = scenarios(opts.distances, opts.probabilities, opts.bases)
            if len(set(cases)) != len(cases):
                raise ValueError("duplicate scenarios in grid")
            rows, pairs = sweep(opts.out, cases, opts.seed, opts.shots, opts.repeats, opts.latency_shots, command)
            print(f"{len(pairs)} paired scenarios; {len(rows)} decoder measurements in {opts.out}")
        elif opts.action == "validate":
            policy = Policy.from_dict(parse_proposal(opts.proposal.read_text(encoding="utf-8-sig")))
            print(json.dumps({"policy_id": policy.policy_id, "proposal": policy.to_dict()}))
        else:
            if opts.action == "search":
                new_campaign(opts.out, opts.strategy, opts.seed, opts.shots, opts.repeats,
                             opts.budget, opts.search_seed, command)
            trials = advance(opts.out, getattr(opts, "proposal", None))
            print(f"{len(trials)} budget slots used; records in {opts.out}")
    except (OSError, ValueError, ValidationError, Timeout) as error:
        parser.exit(2, f"{parser.prog}: {getattr(error, 'message', str(error))}\n")


if __name__ == "__main__":
    main()
