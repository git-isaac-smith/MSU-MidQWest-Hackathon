"""Development, paired evaluation, and replay for the distance-seven decoder."""

import argparse
import csv
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import binomtest

from bascd.stats import paired_counts, paired_ler_difference, wilson95
from local_readout_posterior import LocalReadoutPosterior
from logical_reference import build_reference
from matching_confidence import MatchingConfidence
from stim_data import ROUNDS


PHYSICAL = (7, 0.008, 0.04)
MARGIN_LIMIT = 5.3
CUTOFFS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TIE_TOLERANCE = 1e-12
INITIAL_SHOTS = 150_000
INITIAL_SEED = 83274100
FINAL_SEEDS = (83274101, 83274102, 83274103)
LIST_SEED = 84374100
LIST_FINAL_SEEDS = (84374101, 84374102, 84374103)
LIST_SHOTS = 500_000
LIST_CANDIDATE = {"label": "Matching with alternative corrections", "members": 49, "seed": 84310500}
BLOCK = 8192
BASE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def pack(bits):
    return np.packbits(bits, axis=-1, bitorder="little")


def unpack(packed, count):
    return np.unpackbits(packed, count=count, axis=-1, bitorder="little").view(np.bool_)


def protocol_for(radius, list_matching=False):
    plan = {
        "physical": list(PHYSICAL), "rounds": ROUNDS, "radius": radius,
        "margin_limit": MARGIN_LIMIT, "cutoffs": list(CUTOFFS), "tie_tolerance": TIE_TOLERANCE,
        "development_seed": INITIAL_SEED, "final_seeds": list(FINAL_SEEDS),
        "shots_per_seed": INITIAL_SHOTS, "sample_batch_size": BLOCK,
        "approximation": "At most radius independent readout faults in each of rounds one through four. "
                         "Data transitions sum over all independent data faults. The final data readout is ideal.",
        "routing": "Query on absolute constrained matching cost margin <= 5.3. "
                   "Override MWPM if the alternative parity probability strictly exceeds the cutoff "
                   "and the posterior is outside the tie tolerance. A cutoff of one bypasses querying.",
        "selection": "Minimize development errors over fixed cutoffs. For equal errors choose the largest cutoff.",
        "comparisons": {"primary": "selected versus mwpm", "secondary": "aggressive versus mwpm"},
        "versions": {name: version(name) for name in ("numpy", "pymatching", "scipy", "stim")},
    }
    source = "local_readout_posterior.py"
    if list_matching:
        source = "list_matching.py"
        del plan["radius"]
        plan.update(list_matching=LIST_CANDIDATE.copy(), development_seed=LIST_SEED,
                    final_seeds=list(LIST_FINAL_SEEDS), shots_per_seed=LIST_SHOTS,
                    approximation="Generate correction candidates with ordinary matching and seeded DEM weight "
                                  "perturbations. Sum each unique correction's original DEM probability within "
                                  "its logical class, then normalize across the two classes.")
    plan["source_sha256"] = {name: digest(BASE / name) for name in (
        source, "logical_reference.py", "matching_confidence.py", "stim_data.py",
    )}
    return plan


class DistanceSevenDecoder:
    def __init__(self, radius, *, use_posterior=True, list_matching=None):
        self.reference = build_reference(*PHYSICAL)
        self.confidence = MatchingConfidence(*PHYSICAL) if use_posterior else None
        start = perf_counter()
        self.posterior = None
        self.list_matching = list_matching is not None
        if use_posterior:
            if self.list_matching:
                from list_matching import ListMatchingPosterior

                self.posterior = ListMatchingPosterior(*PHYSICAL, members=list_matching["members"],
                                                       seed=list_matching["seed"])
            else:
                self.posterior = LocalReadoutPosterior(*PHYSICAL, radius=radius)
        self.setup_seconds = perf_counter() - start

    @classmethod
    def from_protocol(cls, plan, *, use_posterior=True):
        return cls(plan.get("radius"), use_posterior=use_posterior, list_matching=plan.get("list_matching"))

    def scores(self, packed):
        records = unpack(packed, self.reference.circuit.num_detectors)
        start = perf_counter()
        mwpm = self.reference.mwpm_parity(records)
        matching_seconds = perf_counter() - start
        route = np.zeros(len(records), dtype=bool)
        if self.posterior is None:
            return mwpm, route, np.empty(0), np.array([matching_seconds, 0, 0])

        start = perf_counter()
        costs = self.confidence.features(records)
        route = np.abs(costs[:, 1] - costs[:, 0]) <= MARGIN_LIMIT
        margin_seconds = perf_counter() - start
        start = perf_counter()
        routed_records = records[route]
        if self.list_matching:
            p_one = self.posterior.predict_proba(routed_records)
        else:
            observed, terminal = self.reference.public_checks(routed_records)
            p_one = self.posterior.predict_proba(observed, terminal)
        posterior_seconds = perf_counter() - start
        return mwpm, route, p_one, np.array([matching_seconds, margin_seconds, posterior_seconds])


def overridden(mwpm, route, p_one, cutoff):
    opposite = np.where(mwpm[route], 1 - p_one, p_one)
    flip = (opposite > cutoff) & (np.abs(p_one - 0.5) > TIE_TOLERANCE)
    result = mwpm.copy()
    result[route] ^= flip
    return result


def archive_path(out, seed):
    return out / f"decisions-{seed}.npz"


def sample_split(decoder, seed, shots, *, path=None, protocol_hash=None, policy_hash=None, cutoff=None):
    sampler = decoder.reference.circuit.compile_detector_sampler(seed=seed)
    detector_count = decoder.reference.circuit.num_detectors
    packed = np.empty((shots, (detector_count + 7) // 8), dtype=np.uint8)
    truth = np.empty(shots, dtype=bool)
    mwpm = np.empty(shots, dtype=bool)
    routed = np.empty(shots, dtype=bool)
    posterior = []
    seconds = np.zeros(3)
    for begin in range(0, shots, BLOCK):
        part = slice(begin, min(begin + BLOCK, shots))
        detectors, observables = sampler.sample(part.stop - part.start, separate_observables=True,
                                                bit_packed=True)
        packed[part] = detectors
        truth[part] = observables[:, 0] & 1
        mwpm[part], routed[part], probability, elapsed = decoder.scores(detectors)
        posterior.append(probability)
        seconds += elapsed
    p_one = np.concatenate(posterior)
    if path is not None:
        arrays = dict(seed=seed, shots=shots, detector_count=detector_count, protocol_sha256=protocol_hash,
                      detectors=packed, truth=pack(truth), mwpm=pack(mwpm), routed=pack(routed),
                      p_one=p_one, seconds=seconds, setup_seconds=decoder.setup_seconds)
        if cutoff is not None:
            arrays["policy_sha256"] = policy_hash
            arrays["aggressive"] = pack(overridden(mwpm, routed, p_one, CUTOFFS[0]))
            arrays["selected"] = pack(overridden(mwpm, routed, p_one, cutoff))
        tmp = path.with_suffix(".partial.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    print(f"Seed {seed}: {shots:,} shots, {len(p_one):,} routed, "
          f"{seconds[0]:.2f} s MWPM, {seconds[1]:.2f} s matching margins, "
          f"{seconds[2]:.2f} s posterior.", flush=True)


def load_split(path, seed, shots, protocol_hash, *, policy_hash=None, cutoff=None):
    with np.load(path, allow_pickle=False) as arrays:
        if (arrays["seed"].item(), arrays["shots"].item(), arrays["protocol_sha256"].item()) != (
                seed, shots, protocol_hash):
            raise ValueError(f"Split does not match the protocol: {path}")
        detector_count = arrays["detector_count"].item()
        detectors = arrays["detectors"]
        if detectors.shape != (shots, (detector_count + 7) // 8) or detectors.dtype != np.uint8:
            raise ValueError(f"Invalid packed detectors: {path}")
        split = {name: unpack(arrays[name], shots) for name in ("truth", "mwpm", "routed")}
        split.update(detector_count=detector_count, detectors=detectors, p_one=arrays["p_one"],
                     seconds=arrays["seconds"], setup_seconds=arrays["setup_seconds"].item())
        p_one = split["p_one"]
        if p_one.shape != (int(split["routed"].sum()),) or not np.all((p_one >= 0) & (p_one <= 1)):
            raise ValueError(f"Invalid posterior probabilities: {path}")
        if cutoff is not None:
            if arrays["policy_sha256"].item() != policy_hash:
                raise ValueError(f"Split does not match the frozen policy: {path}")
            for name, threshold in (("aggressive", CUTOFFS[0]), ("selected", cutoff)):
                split[name] = unpack(arrays[name], shots)
                expected = overridden(split["mwpm"], split["routed"], p_one, threshold)
                if not np.array_equal(split[name], expected):
                    raise ValueError(f"Decision array {name} does not match the frozen rule: {path}")
    return split


def freeze_policy(out, plan, protocol_hash):
    seed, shots = plan["development_seed"], plan["shots_per_seed"]
    path = archive_path(out, seed)
    split = load_split(path, seed, shots, protocol_hash)
    dev_rows = [dict(cutoff=cutoff, errors=int(np.count_nonzero(
        overridden(split["mwpm"], split["routed"], split["p_one"], cutoff) != split["truth"])))
        for cutoff in CUTOFFS]
    selected = min(dev_rows, key=lambda row: (row["errors"], -row["cutoff"]))
    candidate = {key: plan[key] for key in ("radius", "list_matching") if key in plan}
    return dict(protocol_sha256=protocol_hash, development_sha256=digest(path), **candidate,
                cutoff=selected["cutoff"], shots=shots, development=dev_rows,
                routed=int(split["routed"].sum()))


def summarize(out, plan, protocol_hash, policy, policy_hash):
    shots, cutoff = plan["shots_per_seed"], policy["cutoff"]
    by_mode = {name: [] for name in ("mwpm", "aggressive", "selected")}
    files, timings = {}, []
    for seed in plan["final_seeds"]:
        path = archive_path(out, seed)
        split = load_split(path, seed, shots, protocol_hash, policy_hash=policy_hash, cutoff=cutoff)
        files[path.name] = digest(path)
        baseline = split["mwpm"] != split["truth"]
        routed = int(split["routed"].sum())
        for name, rows in by_mode.items():
            failed = split[name] != split["truth"]
            pairs = paired_counts(baseline, failed)
            queries = routed if name == "aggressive" or (name == "selected" and cutoff < 1) else 0
            rows.append(dict(seed=seed, shots=shots, errors=int(failed.sum()), queried=queries,
                             gains=pairs["mwpm_only_failed"], losses=pairs["correlated_only_failed"]))
        matching, margins, posterior = split["seconds"].tolist()
        timings.append(dict(seed=seed, setup_seconds=split["setup_seconds"], mwpm_seconds=matching,
                            margin_seconds=margins, posterior_seconds=posterior))

    rows = []
    for name, by_seed in by_mode.items():
        totals = {key: sum(row[key] for row in by_seed) for key in ("shots", "errors", "queried", "gains", "losses")}
        for row in (totals, *by_seed):
            errors, n = row["errors"], row["shots"]
            row["error_percent"] = 100 * errors / n
            row["wilson95_percent"] = [100 * value for value in wilson95(errors, n)]
            row["query_percent"] = 100 * row["queried"] / n
            if name != "mwpm":
                row["difference_pp"] = 100 * (row["losses"] - row["gains"]) / n
        if name != "mwpm":
            n, wins, losses = totals["shots"], totals["gains"], totals["losses"]
            stats = paired_ler_difference([r["shots"] for r in by_seed], [r["gains"] for r in by_seed],
                                          [r["losses"] for r in by_seed])
            interval = stats["approximate_95_ci"]
            method = "Stratified paired normal approximation"
            if interval is None:
                gain_ci = binomtest(wins, n).proportion_ci(confidence_level=0.975)
                loss_ci = binomtest(losses, n).proportion_ci(confidence_level=0.975)
                interval = (loss_ci.low - gain_ci.high, loss_ci.high - gain_ci.low)
                method = "Conservative exact binomial bounds"
            totals.update(paired95_pp=[100 * value for value in interval], paired95_method=method,
                          mcnemar_p=binomtest(wins, wins + losses).pvalue if wins + losses else 1.0)
        rows.append(dict(mode=name, **totals, per_seed=by_seed))
    return dict(policy_sha256=policy_hash, archives_sha256=files, rows=rows, timings=timings)


def csv_rows(report):
    for row in report["rows"]:
        for part in (row, *row["per_seed"]):
            low, high = part["wilson95_percent"]
            paired_low, paired_high = part.get("paired95_pp", (None, None))
            yield dict(mode=row["mode"], seed=part.get("seed"), shots=part["shots"], errors=part["errors"],
                       error_percent=part["error_percent"], wilson95_low_percent=low, wilson95_high_percent=high,
                       queried=part["queried"], query_percent=part["query_percent"], gains=part["gains"],
                       losses=part["losses"], difference_pp=part.get("difference_pp"),
                       paired95_low_pp=paired_low, paired95_high_pp=paired_high,
                       paired95_method=part.get("paired95_method"), mcnemar_p=part.get("mcnemar_p"))


def run(out, radius, list_matching=False):
    out.mkdir(parents=True, exist_ok=True)
    plan, protocol_path = protocol_for(radius, list_matching), out / "protocol.json"
    if not protocol_path.exists():
        if any(out.iterdir()):
            raise ValueError("Choose an empty study output directory")
        write_json(protocol_path, plan)
    elif read_json(protocol_path) != plan:
        raise ValueError("Current configuration differs from the saved protocol")
    protocol_hash = digest(protocol_path)

    decoder = DistanceSevenDecoder.from_protocol(plan)
    circuit_path = out / "benchmark.stim"
    decoder.reference.circuit.to_file(circuit_path)
    shots, seed = plan["shots_per_seed"], plan["development_seed"]
    path = archive_path(out, seed)
    if not path.exists():
        sample_split(decoder, seed, shots, path=path, protocol_hash=protocol_hash)
    policy = freeze_policy(out, plan, protocol_hash)
    policy_path = out / "policy.json"
    if not policy_path.exists():
        write_json(policy_path, policy)
    elif read_json(policy_path) != policy:
        raise ValueError("Saved policy differs from the development-derived rule")
    candidate = plan["list_matching"]["label"] if list_matching else f"radius {radius}"
    print(f"Policy frozen for {candidate}, cutoff {policy['cutoff']:.1f}.", flush=True)

    policy_hash = digest(policy_path)
    for seed in plan["final_seeds"]:
        path = archive_path(out, seed)
        if not path.exists():
            sample_split(decoder, seed, shots, path=path, protocol_hash=protocol_hash, policy_hash=policy_hash,
                         cutoff=policy["cutoff"])
    report = summarize(out, plan, protocol_hash, policy, policy_hash)
    write_json(out / "results.json", report)
    rows = list(csv_rows(report))
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def verify(out, replay):
    plan, protocol_hash = read_json(out / "protocol.json"), digest(out / "protocol.json")
    policy, policy_hash = read_json(out / "policy.json"), digest(out / "policy.json")
    if freeze_policy(out, plan, protocol_hash) != policy:
        raise ValueError("Stored cutoffs or development counts differ from verified values")
    report = summarize(out, plan, protocol_hash, policy, policy_hash)
    if report != read_json(out / "results.json"):
        raise ValueError("Stored paired summaries differ from verified values")
    with (out / "results.csv").open(encoding="utf-8", newline="") as stream:
        expected = [{key: "" if value is None else str(value) for key, value in row.items()}
                    for row in csv_rows(report)]
        if list(csv.DictReader(stream)) != expected:
            raise ValueError("Stored CSV rows differ from the verified summaries")
    if replay:
        if protocol_for(plan.get("radius"), "list_matching" in plan) != plan:
            raise ValueError("Replay requires the recorded code and dependencies")
        decoder = DistanceSevenDecoder.from_protocol(plan)
        if str(decoder.reference.circuit).strip() != (out / "benchmark.stim").read_text().strip():
            raise ValueError("Stored circuit differs from the replay circuit")
        for seed in (plan["development_seed"], *plan["final_seeds"]):
            replay_split(decoder, out, seed, plan["shots_per_seed"], protocol_hash)
    print("Verified development selection, final decisions, and paired counts.", flush=True)


def replay_split(decoder, out, seed, shots, protocol_hash):
    split = load_split(archive_path(out, seed), seed, shots, protocol_hash)
    sampler = decoder.reference.circuit.compile_detector_sampler(seed=seed)
    previous = 0
    for begin in range(0, shots, BLOCK):
        part = slice(begin, min(begin + BLOCK, shots))
        detectors, labels = sampler.sample(part.stop - part.start, separate_observables=True, bit_packed=True)
        np.testing.assert_array_equal(detectors, split["detectors"][part])
        np.testing.assert_array_equal(labels[:, 0] & 1, split["truth"][part])
        mwpm, route, posterior, _ = decoder.scores(detectors)
        np.testing.assert_array_equal(mwpm, split["mwpm"][part])
        np.testing.assert_array_equal(route, split["routed"][part])
        np.testing.assert_allclose(posterior, split["p_one"][previous:previous + len(posterior)],
                                   rtol=1e-12, atol=1e-14)
        previous += len(posterior)
    print(f"Seed {seed}: replayed {shots:,} records and predictions.", flush=True)


class DeployedDistanceSeven:
    """Apply one locked policy to public packed or unpacked detector records."""

    def __init__(self, folder):
        protocol_path = Path(folder) / "protocol.json"
        plan = read_json(protocol_path)
        policy = read_json(Path(folder) / "policy.json")
        if digest(protocol_path) != policy["protocol_sha256"]:
            raise ValueError("Policy does not match the stored protocol")
        if (plan["physical"] != list(PHYSICAL) or plan["rounds"] != ROUNDS
                or plan["margin_limit"] != MARGIN_LIMIT or plan["tie_tolerance"] != TIE_TOLERANCE):
            raise ValueError("Unsupported distance-seven circuit or routing rule")
        self.threshold = policy["cutoff"]
        if self.threshold not in plan["cutoffs"]:
            raise ValueError("Policy threshold lies outside the declared candidate set")
        self.engine = DistanceSevenDecoder.from_protocol(plan, use_posterior=self.threshold < 1)
        self.ref = self.engine.reference

    def predict_packed(self, records):
        records = np.asarray(records)
        count = self.ref.circuit.num_detectors
        if records.ndim != 2 or records.shape[1] != (count + 7) // 8 or records.dtype != np.uint8:
            raise ValueError("Expected packed uint8 detector records in circuit order")
        mwpm, route, p_one, _ = self.engine.scores(records)
        return overridden(mwpm, route, p_one, self.threshold)

    def predict(self, records, mode="distance7"):
        if mode != "distance7":
            raise ValueError("Expected a distance-seven policy")
        records = np.asarray(records)
        if records.ndim != 2 or records.shape[1] != self.ref.circuit.num_detectors:
            raise ValueError("Expected one column per circuit detector")
        if records.dtype != np.bool_ and (records.dtype.kind not in "bui" or np.any((records != 0) & (records != 1))):
            raise ValueError("Public detector records must be binary")
        return self.predict_packed(pack(records))


def predict(out, records_path, dest):
    decoder = DeployedDistanceSeven(out)
    records = np.load(records_path, allow_pickle=False, mmap_mode="r")
    count = decoder.ref.circuit.num_detectors
    if records.ndim != 2 or records.dtype != np.uint8 or records.shape[1] != (count + 7) // 8:
        raise ValueError("Expected a uint8 .npy array of little-endian packed circuit detector records")
    output = np.empty(len(records), dtype=bool)
    for begin in range(0, len(records), BLOCK):
        part = slice(begin, begin + BLOCK)
        output[part] = decoder.predict_packed(records[part])
    with dest.open("xb") as stream:
        np.save(stream, output, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    study = commands.add_parser("run", help="freeze the development rule and evaluate the three final seeds")
    study.add_argument("--out", type=Path, required=True)
    study.add_argument("--radius", type=int, choices=(1, 2))
    study.add_argument("--candidate", choices=("local-readout", "list-matching"), default="local-readout")
    check = commands.add_parser("verify", help="recount archived decisions and validate the frozen policy")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--replay", action="store_true", help="also regenerate seeded shots and predictions")
    inference = commands.add_parser("predict", help="apply the frozen policy to packed uint8 .npy detectors")
    inference.add_argument("--run", type=Path, required=True)
    inference.add_argument("--detectors", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)
    pilot = commands.add_parser("pilot", help="measure radius cost on a separate timing seed")
    pilot.add_argument("--radius", type=int, choices=(1, 2))
    pilot.add_argument("--candidate", choices=("local-readout", "list-matching"), default="local-readout")
    pilot.add_argument("--shots", type=int, default=8192)
    args = parser.parse_args()
    if args.command in {"run", "pilot"}:
        list_matching = args.candidate == "list-matching"
        if (args.radius is None) != list_matching:
            parser.error("--radius is required for local-readout and omitted for list-matching")
    if args.command == "run":
        run(args.out, args.radius, list_matching)
    elif args.command == "verify":
        verify(args.run, args.replay)
    elif args.command == "predict":
        predict(args.run, args.detectors, args.out)
    else:
        if args.shots < 1:
            parser.error("--shots must be positive")
        plan = protocol_for(args.radius, list_matching)
        decoder = DistanceSevenDecoder.from_protocol(plan)
        print(f"Candidate setup: {decoder.setup_seconds:.2f} s.", flush=True)
        sample_split(decoder, plan["development_seed"] + 9, args.shots)


if __name__ == "__main__":
    main()
