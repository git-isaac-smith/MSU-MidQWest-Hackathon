"""Replay the ten bath stress cells on one million fresh shots each."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
STRESS = EXPERIMENTS / "bath-background-stress-01"
RARE = EXPERIMENTS / "bath-rare-confirmation-02"
ENGINE = EXPERIMENTS / "million-engine-03"
sys.path.insert(0, str(STRESS))
from run import BACKGROUNDS, build_channels, replay, wilson_95
sys.path.insert(0, str(ENGINE))

ARM_IDS = ("fixed_6", "individual")
SHOTS_PER_LAYOUT = 200_000
CHUNK_SHOTS = 10_000
LAYOUTS = 5
STIM_BASE = 628_692_449_100_000
COUNTS = ("shots", "herald_detector_events", "syndrome_detector_events")
OUTCOMES = ("failed", "succeeded")


def hash_json(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hash_file(path):
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def hash_array(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def exact_circuit_hash(circuit):
    instructions = [
        [instruction.name, instruction.tag, [repr(target) for target in instruction.targets_copy()],
         [value.hex() for value in instruction.gate_args_copy()]]
        for instruction in circuit.flattened()
    ]
    return hash_json(instructions)


def write_json(path, value):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def worker_setup(root, arms):
    global UPSTREAM, ARMS
    UPSTREAM = replay.load_upstream(root)
    ARMS = arms


def build_circuits(arm_id, p_check, bath_ids):
    arm = ARMS[arm_id]
    channel_index = {bath_id: index for index, bath_id in enumerate(arm["bath_ids"])}
    draw = [channel_index[bath_id] for bath_id in bath_ids]
    UPSTREAM.P_CHECK = p_check
    rates = UPSTREAM._rates([arm["channels"][index] for index in draw])
    priors = UPSTREAM._rates([arm["priors"][index] for index in draw])
    distance, basis = replay.DISTANCE, replay.BASIS.lower()
    circuit, is_herald = UPSTREAM.build_circuit(distance, basis, *rates, UPSTREAM.P_MEAS)
    prior, is_prior_herald = UPSTREAM.build_circuit(distance, basis, *priors, UPSTREAM.P_MEAS)
    if not np.array_equal(is_herald, is_prior_herald):
        raise ValueError("Physical and prior detector types differ")
    return circuit, prior, is_herald


def layout_plan(record):
    circuit, prior, is_herald = build_circuits(
        record["arm_id"], record["p_check"], record["bath_ids_in_data_qubit_order"]
    )
    return {
        **record,
        "physical_circuit_sha256": exact_circuit_hash(circuit),
        "prior_circuit_sha256": exact_circuit_hash(prior),
        "physical_circuit_stim_text_sha256": sha256(str(circuit).encode()).hexdigest(),
        "prior_circuit_stim_text_sha256": sha256(str(prior).encode()).hexdigest(),
        "herald_mask_sha256": hash_array(is_herald),
    }


def score(detectors, observables, predictions, is_herald):
    failed = np.any(predictions[:, :observables.shape[1]] != observables, axis=1)
    herald_count = np.count_nonzero(detectors[:, is_herald], axis=1)
    syndrome_count = np.count_nonzero(detectors, axis=1) - herald_count
    return {
        label: dict(zip(COUNTS, (int(select.sum()), int(herald_count[select].sum()),
                                 int(syndrome_count[select].sum())), strict=True))
        for label, select in zip(OUTCOMES, (failed, ~failed), strict=True)
    }


def count_totals(rows):
    return {
        outcome: {field: sum(row[outcome][field] for row in rows) for field in COUNTS}
        for outcome in OUTCOMES
    }


def sample(circuit, seed, shots):
    return circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)


def verified_circuits(record):
    circuit, prior, is_herald = build_circuits(
        record["arm_id"], record["p_check"], record["bath_ids_in_data_qubit_order"]
    )
    if (exact_circuit_hash(circuit), exact_circuit_hash(prior), hash_array(is_herald)) != (
        record["physical_circuit_sha256"], record["prior_circuit_sha256"], record["herald_mask_sha256"],
    ):
        raise ValueError(f"Circuit fingerprint differs for {record['id']}")
    return circuit, prior, is_herald


def original_control(record, output, protocol_hash):
    fingerprint = hash_json([protocol_hash, record])
    path = output / "original-controls" / f"{record['id']}.json"
    if path.exists():
        saved = read_json(path)
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"Original-control fingerprint differs in {path}")
        return saved
    circuit, prior, is_herald = verified_circuits(record)
    expected = record["original"]
    detectors, obs = sample(circuit, expected["stim_seed"], expected["shots"])
    decoder = UPSTREAM.ErasureDecoder(prior, is_herald)
    if decoder.skipped:
        raise ValueError(f"Upstream prior has {decoder.skipped} non-graphlike errors")
    start = perf_counter_ns()
    predictions = decoder.decode(detectors)
    elapsed = perf_counter_ns() - start
    counts = score(detectors, obs, predictions, is_herald)
    if counts != {outcome: expected[outcome] for outcome in OUTCOMES}:
        raise ValueError(f"Upstream replay disagrees with saved experiment for {record['id']}")
    result = {
        "fingerprint": fingerprint,
        "stim_seed": expected["stim_seed"],
        "shots": expected["shots"],
        "detector_counts_by_outcome": counts,
        "detectors_sha256": hash_array(detectors),
        "predictions_sha256": hash_array(predictions),
        "decode_ns_under_campaign_load": elapsed,
    }
    write_json(path, result)
    return result


def fresh_layout(record, output, protocol_hash, engine_hash, smoke):
    from engine import FastErasureDecoder

    fingerprint = hash_json([protocol_hash, engine_hash, record, smoke])
    path = output / "checkpoints" / f"{record['id']}.json"
    chunks = []
    if path.exists():
        saved = read_json(path)
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"Fresh checkpoint fingerprint differs in {path}")
        chunks = saved["chunks"]
    expected_seeds = [record["stim_seeds"][0] + 10**12] if smoke else record["stim_seeds"]
    expected_shots = 250 if smoke else CHUNK_SHOTS
    if len(chunks) > len(expected_seeds):
        raise ValueError(f"Unexpected chunk count in {path}")
    for chunk, seed in zip(chunks, expected_seeds):
        counts = chunk["detector_counts_by_outcome"]
        if chunk["stim_seed"] != seed or chunk["shots"] != expected_shots or (
            sum(counts[outcome]["shots"] for outcome in OUTCOMES) != expected_shots
        ) or chunk["logical_failures"] != counts["failed"]["shots"]:
            raise ValueError(f"Inconsistent chunk in {path}")
    if len(chunks) == len(expected_seeds):
        return saved

    circuit, prior, is_herald = verified_circuits(record)
    control = original_control(record, output, protocol_hash)
    decoder = FastErasureDecoder(UPSTREAM, prior, is_herald)
    detectors, obs = sample(circuit, control["stim_seed"], control["shots"])
    predictions = decoder.decode(detectors)
    if hash_array(detectors) != control["detectors_sha256"] or (
        hash_array(predictions) != control["predictions_sha256"]
    ):
        raise ValueError(f"Fast predictions differ from upstream for {record['id']}")

    for index, seed in enumerate(expected_seeds[len(chunks):], start=len(chunks)):
        start = perf_counter_ns()
        detectors, obs = sample(circuit, seed, expected_shots)
        sampled = perf_counter_ns()
        predictions = decoder.decode(detectors)
        decoded = perf_counter_ns()
        counts = score(detectors, obs, predictions, is_herald)
        if index == 0:
            reference = UPSTREAM.ErasureDecoder(prior, is_herald)
            fresh_check = min(128, expected_shots)
            if not np.array_equal(reference.decode(detectors[:fresh_check]), predictions[:fresh_check]):
                raise ValueError(f"Fresh predictions differ from upstream for {record['id']}")
        chunks.append({
            "chunk_index": index,
            "stim_seed": seed,
            "shots": expected_shots,
            "logical_failures": counts["failed"]["shots"],
            "detector_counts_by_outcome": counts,
            "detectors_sha256": hash_array(detectors),
            "observables_sha256": hash_array(obs),
            "predictions_sha256": hash_array(predictions),
            "sample_ns_under_campaign_load": sampled - start,
            "decode_ns_under_campaign_load": decoded - sampled,
        })
        totals = count_totals([chunk["detector_counts_by_outcome"] for chunk in chunks])
        shots = sum(totals[outcome]["shots"] for outcome in OUTCOMES)
        failures = totals["failed"]["shots"]
        saved = {
            "fingerprint": fingerprint,
            **{key: value for key, value in record.items() if key not in ("original", "stim_seeds")},
            "shots": shots,
            "logical_failures": failures,
            "logical_error_rate": failures / shots,
            "wilson_95": wilson_95(failures, shots),
            "detector_counts_by_outcome": totals,
            "original_seed_control": {**control, "matching_prediction_bits": control["shots"]},
            "fresh_upstream_prediction_audit_shots": min(128, expected_shots),
            "chunks": chunks,
        }
        write_json(path, saved)
    return saved


def pool_jobs(worker, pool, records, *args):
    jobs = {pool.submit(worker, record, *args): record for record in records}
    results = {}
    for job in as_completed(jobs):
        record = jobs[job]
        row = job.result()
        results[record["id"]] = row
        print(f"{worker.__name__} {record['id']}: {row.get('logical_failures', 'verified')}/{row['shots']}",
              flush=True)
    return results


def make_plan(root):
    revision = git_output(root, "rev-parse", "HEAD")
    if revision != replay.UPSTREAM_COMMIT or git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Use a clean checkout at the pinned upstream revision")
    source = read_json(STRESS / "result.json")
    rare = read_json(RARE / "result.json")
    upstream = replay.load_upstream(root)
    versions = {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "scipy", "stim", "pymatching", "pandas", "qutip")
    }}
    if versions != source["versions"]:
        raise ValueError(f"Use the original seed replay package versions. Found {versions}")
    if source["p_measurement_flip"] != upstream.P_MEAS or upstream.N_DRAWS != LAYOUTS:
        raise ValueError("Original measurement error or layout count differs from upstream")
    arms = {arm_id: build_channels(root, upstream, arm) for arm_id, arm in zip(ARM_IDS, replay.ARMS, strict=True)}
    layout_seeds = [upstream.SEED + 1000 * replay.DISTANCE + j for j in range(LAYOUTS)]
    bath_ids = np.asarray(arms[ARM_IDS[0]]["bath_ids"])
    layouts = [bath_ids[np.random.default_rng(seed).integers(0, len(bath_ids), replay.DISTANCE**2)].tolist()
               for seed in layout_seeds]
    if layout_seeds != source["layout_seeds"] or layouts != source["layout_bath_ids_in_data_qubit_order"]:
        raise ValueError("Pinned upstream layouts differ from the original stress layouts")
    if {(point["arm"], point["p_check"]) for point in source["points"]} != set(
        (arm, p_check) for arm in replay.ARMS for p_check in BACKGROUNDS
    ):
        raise ValueError("Stress cell set differs from expected five levels and two arms")

    records = []
    prior_seeds = {chunk["stim_seed"] for layout in rare["layouts"] for chunk in layout["chunks"]}
    for cell_index, point in enumerate(source["points"]):
        arm_id = ARM_IDS[replay.ARMS.index(point["arm"])]
        for i, (seed, layout, control) in enumerate(zip(layout_seeds, layouts, point["layout_counts"], strict=True)):
            prior_seeds.add(control["stim_seed"])
            records.append({
                "id": f"{arm_id}-p{point['p_check']:g}-layout{i + 1:02d}",
                "arm_id": arm_id,
                "arm": point["arm"],
                "p_check": point["p_check"],
                "layout_index": i + 1,
                "layout_seed": seed,
                "bath_ids_in_data_qubit_order": layout,
                "stim_seeds": [STIM_BASE + 1000 * cell_index + 100 * i + j
                               for j in range(SHOTS_PER_LAYOUT // CHUNK_SHOTS)],
                "original": {"shots": source["shots_per_layout"], **control},
            })
    fresh_seeds = [seed for record in records for seed in record["stim_seeds"]]
    if len(fresh_seeds) != len(set(fresh_seeds)) or prior_seeds.intersection(fresh_seeds):
        raise ValueError("Fresh seed overlap")

    worker_setup(root, arms)
    plans = [layout_plan(record) for record in records]
    metadata = {
        **{key: source[key] for key in (
            "upstream_repository", "upstream_commit", "upstream_source", "upstream_table_directory", "attachment",
            "gamma_sys", "clock_period", "distance", "basis", "syndrome_rounds", "native_p_check",
            "p_measurement_flip", "p_reset_flip", "layout_seeds", "layout_bath_ids_in_data_qubit_order",
        )},
        "format_version": 1,
        "shots_per_layout": SHOTS_PER_LAYOUT,
        "shots_per_cell": SHOTS_PER_LAYOUT * LAYOUTS,
        "chunk_shots": CHUNK_SHOTS,
        "stim_seed_rule": f"{STIM_BASE} + 1000 * zero-based stress cell + 100 * zero-based layout + zero-based chunk",
        "interval_scope": "Monte Carlo precision conditional on the five fixed layouts",
        "environment": {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "versions": versions},
        "input_sha256_normalized_lf": {
            str(path.relative_to(EXPERIMENTS)): hash_file(path) for path in (
                Path(__file__), STRESS / "result.json", STRESS / "run.py", RARE / "result.json",
                EXPERIMENTS / "sakidja-table5-replay-01" / "replay.py",
            )
        },
        "upstream_input_sha256_normalized_lf": {
            str(path.as_posix()): hash_file(root / path) for path in (
                replay.SOURCE, replay.TABLE_DIR / "raw_parameters_per_bath.csv",
                replay.TABLE_DIR / "raw_round_channels_per_bath.csv",
            )
        },
        "arms": [{"arm_id": arm_id, **{key: value for key, value in row.items()
                                        if key not in ("bath_ids", "channels", "priors")}}
                 for arm_id, row in arms.items()],
    }
    return arms, {**metadata, "protocol_sha256": hash_json([metadata, plans]), "layouts": plans}


def aggregate(plan, rows, engine_hash):
    points = []
    for start in range(0, len(plan["layouts"]), LAYOUTS):
        records = plan["layouts"][start:start + LAYOUTS]
        layouts = [rows[record["id"]] for record in records]
        if any(layout["shots"] != SHOTS_PER_LAYOUT for layout in layouts):
            raise ValueError("Layout shot count differs from planned sample size")
        counts = count_totals([layout["detector_counts_by_outcome"] for layout in layouts])
        shots = sum(value["shots"] for value in counts.values())
        failures = counts["failed"]["shots"]
        points.append({
            **{key: layouts[0][key] for key in ("arm_id", "arm", "p_check")},
            "shots": shots,
            "logical_failures": failures,
            "logical_error_rate": failures / shots,
            "wilson_95": wilson_95(failures, shots),
            "detector_means_by_outcome": {
                outcome: {f"mean_{kind}_detector_events_per_shot":
                          fields[f"{kind}_detector_events"] / fields["shots"] if fields["shots"] else None
                          for kind in ("herald", "syndrome")}
                for outcome, fields in counts.items()
            },
            "layouts": layouts,
        })
    return {**{key: value for key, value in plan.items() if key != "layouts"},
            "engine_source_sha256_normalized_lf": engine_hash, "points": points}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=("plan", "original-controls", "run", "smoke"), default="run")
    parser.add_argument("--arm", choices=ARM_IDS)
    parser.add_argument("--p-check", type=float, choices=BACKGROUNDS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between one and four")
    if args.mode == "smoke" and args.output.resolve() == HERE:
        parser.error("Provide a separate output directory for smoke testing")
    root, output = args.upstream.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "original-controls").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    arms, plan = make_plan(root)
    write_json(output / "plan.json", plan)
    print(f"Verified plan with {len(plan['layouts'])} layouts", flush=True)
    if args.mode == "plan":
        return

    records = [record for record in plan["layouts"] if (args.arm is None or record["arm_id"] == args.arm)
               and (args.p_check is None or record["p_check"] == args.p_check)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_setup, initargs=(root, arms)) as pool:
        if args.mode == "original-controls":
            pool_jobs(original_control, pool, records, output, plan["protocol_sha256"])
            return
        engine_hash = {name: hash_file(ENGINE / name) for name in ("engine.py", "engine.cpp", "build_engine.py")}
        rows = pool_jobs(fresh_layout, pool, records, output, plan["protocol_sha256"],
                         engine_hash, args.mode == "smoke")

    if args.mode == "run" and len(rows) == len(plan["layouts"]):
        write_json(output / "result.json", aggregate(plan, rows, engine_hash))


if __name__ == "__main__":
    main()
