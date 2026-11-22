"""Score one million fresh shots per frozen bath timing policy."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, perf_counter_ns, sleep

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "bath-clock-controls-02"
BATH = HERE.parent / "million-bath-03"
sys.path.insert(0, str(PREVIOUS))
import fft_replay
from fft_replay import channels_for_run, load_pinned, replay
from run import ARMS as CLOCK_ARMS, BASIS, CHANNEL_KEYS, CLOCK, DISTANCE, INDIVIDUAL, describe_times


ARMS = (*CLOCK_ARMS, *fft_replay.ARMS)
SCORED_ARMS = ("global", "global_installed_calibration", "shuffled", "dominant_tone_timing", "fft_only")
CHUNK = 10_000
SHOTS = 1_000_000
LAYOUT_SHOTS = 200_000
SEED_NAMESPACE = 0x510C


def digest(data):
    return sha256(data).hexdigest()


def array_digest(array):
    return digest(np.ascontiguousarray(array).tobytes())


def read(path):
    return json.loads(path.read_bytes())


def read_shared_checkpoint(path):
    if sys.platform != "win32":
        return read(path)
    import _winapi
    import msvcrt

    share_read_write_delete = 0x1 | 0x2 | 0x4
    handle = _winapi.CreateFile(str(path), _winapi.GENERIC_READ, share_read_write_delete,
                                0, _winapi.OPEN_EXISTING, 0, 0)
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "rb") as checkpoint:
        return json.load(checkpoint)


def write(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            sleep(0.05 * (2**attempt))


def exact_circuit_hash(circuit):
    instructions = [[instruction.name, instruction.tag,
                     [repr(target) for target in instruction.targets_copy()],
                     [value.hex() for value in instruction.gate_args_copy()]]
                    for instruction in circuit.flattened()]
    return digest(json.dumps(instructions, separators=(",", ":")).encode())


def circuit_fingerprints(physical, prior):
    return {f"{name}_circuit_sha256": digest(str(circuit).encode())
            for name, circuit in (("physical", physical), ("prior", prior))} | {
        f"{name}_exact_circuit_sha256": exact_circuit_hash(circuit)
        for name, circuit in (("physical", physical), ("prior", prior))}


def physical_differences(first, second):
    if len(first) != len(second):
        raise ValueError("The two circuits have different instructions")
    deltas = []
    for left, right in zip(first, second, strict=True):
        if (left.name, left.tag, left.targets_copy()) != (right.name, right.tag, right.targets_copy()):
            raise ValueError("The two circuits have different instructions")
        deltas.extend(abs(a - b) for a, b in zip(left.gate_args_copy(), right.gate_args_copy(), strict=True))
    return {"different_gate_arguments": sum(delta != 0 for delta in deltas),
            "max_abs_gate_argument_difference": max(deltas, default=0.0)}


def prepare(root):
    upstream, revision, parameter_path, channel_path = load_pinned(root)
    paths = {"control_calibration": PREVIOUS / "calibration.json",
             "fft_calibration": PREVIOUS / "fft_calibration.json",
             "fft_predictions": HERE.parent / "bath-fft-timing-02" / "per_bath.csv",
             "control_result": PREVIOUS / "result.json", "fft_result": PREVIOUS / "fft_result.json"}
    frozen_hashes = {name: digest(path.read_bytes()) for name, path in paths.items()}
    calibration = read(paths["control_calibration"])
    fft = read(paths["fft_calibration"])
    if (calibration["upstream_commit"] != revision or fft["upstream_commit"] != revision
            or calibration["parameter_csv_sha256"] != digest(parameter_path.read_bytes())
            or fft["control_calibration_sha256"] != frozen_hashes["control_calibration"]
            or fft["fft_predictions_sha256"] != frozen_hashes["fft_predictions"]):
        raise ValueError("Frozen inputs do not match the upstream model or each other")

    layouts = calibration["frozen_layouts"]
    if len(layouts) * LAYOUT_SHOTS != SHOTS:
        raise ValueError("Unexpected number of frozen layouts")
    schedules = {arm: {int(bath): tau for bath, tau in times.items()}
                 for arm, times in fft["times_by_arm_and_bath"].items()}
    for layout in layouts:
        for arm, times in schedules.items():
            layout["times"][arm] = [times[bath] for bath in layout["bath_ids"]]
    params = {int(row["Bath"]): row for row in replay.read_rows(parameter_path) if row["Set"] == "unseen"}
    published_rows = [row for row in replay.read_rows(channel_path) if float(row["T_clock"]) == CLOCK]
    published = {arm: {int(row["Bath"]): row for row in published_rows if row["Arm"] == label}
                 for arm, label in (("fixed_6", upstream.ARM_CLOCK), ("individual", INDIVIDUAL))}
    fft_inputs = {int(bath): value for bath, value in fft["fft_frame_and_prior_by_bath"].items()}
    historical = {arm["arm"]: arm for key in ("control_result", "fft_result") for arm in read(paths[key])["arms"]}
    if len({historical[arm]["shots"] for arm in ARMS}) != 1:
        raise ValueError("Unexpected historical shot totals")
    previous_seeds = {row["stim_seed"] for arm in historical.values() for row in arm["layouts"]}

    circuits, information = {}, {}
    for arm in ARMS:
        if arm == "fft_only":
            pairs, _ = fft_replay.fft_channels(upstream, schedules[arm], fft_inputs, params)
        else:
            pairs, _ = channels_for_run(upstream, arm, layouts, params, published.get(arm))
        circuits[arm], information[arm] = [], []
        for index, layout in enumerate(layouts):
            times = layout["times"][arm]
            physical, priors = zip(*(pairs[site] for site in zip(layout["bath_ids"], times, strict=True)), strict=True)
            circuit, flags = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(physical), upstream.P_MEAS)
            prior, prior_flags = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(priors), upstream.P_MEAS)
            if not np.array_equal(flags, prior_flags):
                raise ValueError(f"Herald structure differs for {arm}, layout {index}")
            fingerprints = circuit_fingerprints(circuit, prior)
            historical_row = historical[arm]["layouts"][index]
            for key in ("physical_circuit_sha256", "prior_circuit_sha256"):
                if fingerprints[key] != historical_row[key]:
                    raise ValueError(f"Historical {key} differs for {arm}, layout {index}")
            circuits[arm].append((circuit, prior, flags))
            information[arm].append({"layout_index": index, "layout_seed": calibration["layout_seeds"][index],
                                     "bath_ids": layout["bath_ids"], **fingerprints,
                                     "herald_mask_sha256": array_digest(flags),
                                     "time_distribution": describe_times(times),
                                     "mean_physical_channel": {key: float(np.mean([c[key] for c in physical]))
                                                               for key in CHANNEL_KEYS},
                                     "mean_prior_channel": {key: float(np.mean([c[key] for c in priors]))
                                                            for key in CHANNEL_KEYS}})

    for index, (fit, fft_timing) in enumerate(zip(circuits["individual"], circuits["fft_timing"], strict=True)):
        if fit[:2] != fft_timing[:2] or not np.array_equal(fit[2], fft_timing[2]):
            raise ValueError(f"Individual and FFT timing differ on layout {index}")
    coupling = [physical_differences(fit[0], fft[0])
                for fit, fft in zip(circuits["individual"], circuits["fft_only"], strict=True)]
    info = {"upstream_commit": revision, "upstream_parameter_csv_sha256": digest(parameter_path.read_bytes()),
            "upstream_channel_csv_sha256": digest(channel_path.read_bytes()), "frozen_inputs_sha256": frozen_hashes,
            "versions": {"python": platform.python_version(), **{pkg: version(pkg) for pkg in
                         ("numpy", "scipy", "pandas", "qutip", "stim", "pymatching")}},
            "distance": DISTANCE, "basis": BASIS, "rounds": DISTANCE, "attachment": upstream.ATTACH,
            "gamma_sys": 0.0, "p_check": upstream.P_CHECK, "p_meas": upstream.P_MEAS,
            "shots_per_arm": SHOTS, "shots_per_layout": LAYOUT_SHOTS, "chunk_shots": CHUNK,
            "fft_only_vs_individual_physical_instruction_differences_by_layout": coupling}
    for arm in SCORED_ARMS:
        for index in range(len(layouts)):
            if any(seed_for(arm, index, chunk) in previous_seeds for chunk in range(LAYOUT_SHOTS // CHUNK)):
                raise ValueError("Fresh seed overlaps the previous timing controls")
    return upstream, layouts, information, circuits, info


def seed_for(arm, layout_index, chunk_index):
    label = f"million-controls-03/{arm}/{layout_index}/{chunk_index}".encode()
    return (SEED_NAMESPACE << 48) | int.from_bytes(sha256(label).digest()[:6], "big")


def chunk_path(arm, layout_index, chunk_index, paired_mode=False):
    directory = "paired-chunks" if paired_mode else "chunks"
    return HERE / directory / arm / f"layout-{layout_index:02d}" / f"chunk-{chunk_index:02d}.json"


def specifications(arm, layout_index, chunk_index, metadata, versions, engine_hash, seed=None, shots=CHUNK):
    return {"arm": arm, "layout_index": layout_index, "chunk_index": chunk_index,
            "shots": shots, "stim_seed": seed_for(arm, layout_index, chunk_index) if seed is None else seed,
            "physical_exact_circuit_sha256": metadata["physical_exact_circuit_sha256"],
            "prior_exact_circuit_sha256": metadata["prior_exact_circuit_sha256"],
            "versions": {key: versions[key] for key in ("stim", "pymatching", "numpy")},
            "decoder_engine_sha256": engine_hash}


def validate_chunk(row, expected):
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing chunk metadata differs for {expected['arm']} layout {expected['layout_index']} "
                         f"chunk {expected['chunk_index']}")
    if not 0 <= row["logical_errors"] <= row["shots"]:
        raise ValueError("Chunk logical error count is outside the shot count")
    return row


def native_record(arm, metadata):
    index = metadata["layout_index"]
    path = BATH / "checkpoints" / f"{arm}-p0.001-layout{index + 1:02d}.json"
    try:
        row = read_shared_checkpoint(path)
    except FileNotFoundError:
        return None, path
    layout_fields = {"arm_id": arm, "p_check": 0.001, "layout_index": index + 1,
                     "layout_seed": metadata["layout_seed"], "bath_ids_in_data_qubit_order": metadata["bath_ids"],
                     "physical_circuit_sha256": metadata["physical_exact_circuit_sha256"],
                     "prior_circuit_sha256": metadata["prior_exact_circuit_sha256"],
                     "herald_mask_sha256": metadata["herald_mask_sha256"]}
    if any(row.get(key) != value for key, value in layout_fields.items()):
        raise ValueError(f"Native bath and controls circuits or layouts differ in {path}")
    if row["shots"] != sum(chunk["shots"] for chunk in row["chunks"]) or row["logical_failures"] != sum(
            chunk["logical_failures"] for chunk in row["chunks"]):
        raise ValueError(f"Native bath chunk totals differ in {path}")
    return row, path


def detector_sample(circuit, seed, shots):
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    packed_obs = np.packbits(obs).tobytes()
    return dets, obs, digest(np.packbits(dets).tobytes() + packed_obs), digest(packed_obs)


def verify_decoder(upstream, arm, index, circuit_data, dets, obs, metadata, engine, engine_hash):
    path = HERE / "decoder-validation" / arm / f"layout-{index:02d}.json"
    expected = {"arm": arm, "layout_index": index, "shots": len(dets),
                "prior_exact_circuit_sha256": metadata["prior_exact_circuit_sha256"],
                "test_sample_sha256": digest(np.packbits(dets).tobytes() + np.packbits(obs).tobytes()),
                "decoder_engine_sha256": engine_hash}
    saved = read(path) if path.exists() else {}
    if all(saved.get(key) == value for key, value in expected.items()) and saved["prediction_mismatches"] == 0:
        return
    _, prior, flags = circuit_data
    reference = upstream.ErasureDecoder(prior, flags)
    compiled = engine.FastErasureDecoder(upstream, prior, flags)
    expected_prediction = reference.decode(dets)[..., :obs.shape[1]]
    actual_prediction = compiled.decode(dets)[..., :obs.shape[1]]
    mismatches = int(np.count_nonzero(np.any(actual_prediction != expected_prediction, axis=1)))
    if mismatches:
        raise ValueError(f"Compiled decoder differs on {mismatches} author predictions for {arm} layout {index}")
    write(path, {**expected, "prediction_mismatches": mismatches,
                 "prediction_sha256": digest(np.packbits(expected_prediction).tobytes())})


def run_layout(upstream, arm, layout_index, circuit_data, metadata, versions, engine,
               paired=None, fitted_decoder=None):
    circuit, prior, flags = circuit_data
    engine_hash = digest(Path(engine.__file__).read_bytes())
    decoder = engine.FastErasureDecoder(upstream, prior, flags)
    rows = []
    batches = [None] * (LAYOUT_SHOTS // CHUNK) if paired is None else paired
    for chunk, counterpart in enumerate(batches):
        path = chunk_path(arm, layout_index, chunk, paired_mode=counterpart is not None)
        expected = specifications(arm, layout_index, chunk, metadata, versions, engine_hash,
                                  seed=None if counterpart is None else counterpart["stim_seed"],
                                  shots=CHUNK if counterpart is None else counterpart["shots"])
        if counterpart is not None:
            expected.update(paired_detectors_sha256=counterpart["detectors_sha256"],
                            paired_observables_sha256=counterpart["observables_sha256"],
                            paired_fitted_predictions_sha256=counterpart["predictions_sha256"])
        if chunk == 0:
            dets, obs, _, _ = detector_sample(circuit, expected["stim_seed"], expected["shots"])
            verify_decoder(upstream, arm, layout_index, circuit_data, dets[:1024], obs[:1024], metadata, engine, engine_hash)
        if path.exists():
            rows.append(validate_chunk(read(path), expected))
            continue
        start = perf_counter()
        dets, obs, sample_hash, obs_hash = detector_sample(circuit, expected["stim_seed"], expected["shots"])
        if counterpart is not None and (array_digest(dets) != counterpart["detectors_sha256"]
                                        or array_digest(obs) != counterpart["observables_sha256"]):
            raise ValueError(f"Paired FFT-only sample differs from fitted layout {layout_index}, chunk {chunk}. "
                             "Use --fft-independent for a fully independent draw.")
        prediction = decoder.decode(dets)[..., :obs.shape[1]]
        failures = np.any(prediction != obs, axis=1)
        row = {**expected, "logical_errors": int(np.count_nonzero(failures)),
               "sample_sha256": sample_hash,
               "observable_sha256": obs_hash, "prediction_sha256": digest(np.packbits(prediction).tobytes()),
               "failure_mask_sha256": digest(np.packbits(failures).tobytes()),
               "elapsed_seconds": perf_counter() - start}
        if counterpart is not None:
            fft_prediction_hash = array_digest(prediction)
            if fft_prediction_hash == counterpart["predictions_sha256"]:
                fitted_failures = failures
            else:
                fitted_prediction = fitted_decoder.decode(dets)[..., :obs.shape[1]]
                if array_digest(fitted_prediction) != counterpart["predictions_sha256"]:
                    raise ValueError("Fitted predictions differ from the verified native records")
                fitted_failures = np.any(fitted_prediction != obs, axis=1)
            if np.count_nonzero(fitted_failures) != counterpart["logical_failures"]:
                raise ValueError("Paired fitted failures differ from the native totals")
            row.update(fft_predictions_sha256=fft_prediction_hash,
                       paired_errors={"both_fail": int(np.count_nonzero(failures & fitted_failures)),
                                      "fft_only_fails": int(np.count_nonzero(failures & ~fitted_failures)),
                                      "fitted_only_fails": int(np.count_nonzero(~failures & fitted_failures)),
                                      "neither_fails": int(np.count_nonzero(~failures & ~fitted_failures))})
        write(path, row)
        rows.append(row)
        print(f"{arm} layout {layout_index + 1}, chunk {chunk + 1}: {row['logical_errors']}/{row['shots']}", flush=True)
    return rows


def relative(path):
    return Path(os.path.relpath(path, HERE)).as_posix()


def source_reference(path):
    return {"path": relative(path), "sha256": digest(path.read_bytes())}


def benchmark(upstream, circuits, information, info, engine):
    timing_shots, repetitions = 512, 5
    arms = tuple(arm for arm in ARMS if arm != "fft_timing")
    keys = [(arm, index) for arm in arms for index in range(len(circuits[arm]))]
    keys = [keys[index] for index in np.random.default_rng(312051411).permutation(len(keys))]
    measured = {}
    for arm, index in keys:
        physical, prior, flags = circuits[arm][index]
        seed = (0x510D << 48) | (ARMS.index(arm) << 16) | index
        dets, _, sample_hash, _ = detector_sample(physical, seed, timing_shots)
        decoder = engine.FastErasureDecoder(upstream, prior, flags)
        decoder.decode(dets)
        elapsed_ns = []
        for _ in range(repetitions):
            start = perf_counter_ns()
            decoder.decode(dets)
            elapsed_ns.append(perf_counter_ns() - start)
        measured[(arm, index)] = {"layout_index": index, "stim_seed": seed, "sample_sha256": sample_hash,
                                  "shots": timing_shots, "elapsed_ns": elapsed_ns,
                                  "median_ns": int(np.median(elapsed_ns)),
                                  **{key: information[arm][index][key] for key in
                                     ("physical_exact_circuit_sha256", "prior_exact_circuit_sha256")}}
    rows = []
    for arm in ARMS:
        timing_arm = "individual" if arm == "fft_timing" else arm
        layouts = [measured[(timing_arm, index)] for index in range(len(circuits[arm]))]
        us_per_shot = float(np.mean([row["median_ns"] for row in layouts])) / timing_shots / 1_000
        rows.append({"arm": arm, "us_per_shot": us_per_shot, "us_per_round": us_per_shot / DISTANCE,
                     "reused_timing_from": timing_arm if timing_arm != arm else None, "layouts": layouts})
        print(f"{arm}: {us_per_shot:.3f} us/shot", flush=True)
    write(HERE / "accelerated-timing.json", {
        "protocol": "Inference-only decoder.decode wall time under shared machine load. The timed calls include per-batch "
                    "native matching graph initialization. Stim sampling, decoder construction, warm-up, and result "
                    "scoring took place outside the timed intervals.",
        "estimator": "Mean of the five layout median times, divided by shots and optionally by syndrome rounds.",
        "shots_per_layout_batch": timing_shots, "warmup_calls_per_layout": 1, "timed_calls_per_layout": repetitions,
        "rounds": DISTANCE, "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "versions": info["versions"],
        "decoder_engine_source_sha256": digest(Path(engine.__file__).read_bytes()),
        "timed_layout_order": [{"arm": arm, "layout_index": index} for arm, index in keys], "arms": rows,
    })


def aggregate(fft_independent):
    manifest = read(HERE / "manifest.json")
    information = manifest.pop("layout_metadata_by_arm")
    engine_path = HERE.parent / "million-engine-03"
    engine_hash = digest((engine_path / "engine.py").read_bytes())
    arms, unique_seeds, unique_samples = [], set(), set()
    for arm in ARMS:
        layout_rows = []
        for metadata in information[arm]:
            index = metadata["layout_index"]
            if arm in ("fixed_6", "individual", "fft_timing"):
                native_arm = "individual" if arm == "fft_timing" else arm
                native, path = native_record(native_arm, metadata)
                if native is None or native["shots"] != LAYOUT_SHOTS:
                    raise ValueError(f"Native record is incomplete in {path}")
                chunks = native["chunks"]
                source = [source_reference(path)]
                same_evaluations = "individual" if arm == "fft_timing" else "bath background p_check=0.001"
                failures, audit = native["logical_failures"], native["original_seed_control"]["matching_prediction_bits"]
                recorded_sample_key = "detectors_sha256"
            else:
                paired = arm == "fft_only" and not fft_independent
                counterpart = None
                if paired:
                    native, _ = native_record("individual", information["individual"][index])
                    if native is None or native["shots"] != LAYOUT_SHOTS:
                        raise ValueError(f"Incomplete native fitted counterpart for layout {index}")
                    counterpart = native["chunks"]
                chunks, source = [], []
                for batch in range(LAYOUT_SHOTS // CHUNK):
                    path = chunk_path(arm, index, batch, paired_mode=paired)
                    expected = specifications(arm, index, batch, metadata, manifest["versions"], engine_hash,
                                              seed=counterpart[batch]["stim_seed"] if paired else None)
                    row = validate_chunk(read(path), expected)
                    if paired:
                        if (row["paired_detectors_sha256"] != counterpart[batch]["detectors_sha256"]
                                or row["paired_observables_sha256"] != counterpart[batch]["observables_sha256"]
                                or row["paired_fitted_predictions_sha256"] != counterpart[batch]["predictions_sha256"]):
                            raise ValueError(f"Paired record digest differs in {path}")
                        counts = row["paired_errors"]
                        if (sum(counts.values()) != row["shots"] or min(counts.values()) < 0
                                or counts["both_fail"] + counts["fft_only_fails"] != row["logical_errors"]
                                or counts["both_fail"] + counts["fitted_only_fails"] != counterpart[batch]["logical_failures"]):
                            raise ValueError(f"Paired failure counts differ in {path}")
                    chunks.append(row)
                    source.append(source_reference(path))
                failures = sum(chunk["logical_errors"] for chunk in chunks)
                audit_path = HERE / "decoder-validation" / arm / f"layout-{index:02d}.json"
                validated = read(audit_path)
                if (validated["prediction_mismatches"] or validated["decoder_engine_sha256"] != engine_hash
                        or validated["prior_exact_circuit_sha256"] != metadata["prior_exact_circuit_sha256"]):
                    raise ValueError(f"Decoder validation does not match {arm} layout {index}")
                audit = validated["shots"]
                same_evaluations = "individual (shared verified samples, each prior scored separately)" if paired else None
                recorded_sample_key = "sample_sha256"
            shots = sum(chunk["shots"] for chunk in chunks)
            if shots != LAYOUT_SHOTS:
                raise ValueError(f"Unexpected shot count for {arm} layout {index}")
            row_seeds = {chunk["stim_seed"] for chunk in chunks}
            row_samples = {chunk[recorded_sample_key] for chunk in chunks}
            if len(row_seeds) != len(chunks) or len(row_samples) != len(chunks):
                raise ValueError(f"Duplicate seeds or samples within {arm} layout {index}")
            if arm == "fft_timing" or (arm == "fft_only" and not fft_independent):
                if not row_seeds <= unique_seeds:
                    raise ValueError(f"Paired seeds were absent from individual for {arm} layout {index}")
            else:
                if row_seeds & unique_seeds or row_samples & unique_samples:
                    raise ValueError(f"Independent seeds or samples overlap for {arm} layout {index}")
                unique_seeds.update(row_seeds)
                unique_samples.update(row_samples)
            layout_row = {**metadata, "shots": shots, "logical_errors": failures,
                          "stim_seeds": [chunk["stim_seed"] for chunk in chunks],
                          "score_records": source, "author_decoder_audit_shots": audit,
                          "shared_samples_with": same_evaluations}
            if arm == "fft_only" and not fft_independent:
                layout_row["paired_errors_vs_individual"] = {key: sum(chunk["paired_errors"][key] for chunk in chunks)
                                                             for key in chunks[0]["paired_errors"]}
            layout_rows.append(layout_row)

        failures = sum(row["logical_errors"] for row in layout_rows)
        times = [float(tau) for row in layout_rows for tau, count in row["time_distribution"]["count_by_time"].items()
                 for _ in range(count)]
        arm_row = {"arm": arm, "shots": SHOTS, "logical_errors": failures, "ler": failures / SHOTS,
                   "conditional_binomial_clopper_pearson_95": replay.interval_95(failures, SHOTS),
                   "time_distribution": describe_times(times), "layouts": layout_rows,
                   "mean_physical_channel": {key: float(np.mean([row["mean_physical_channel"][key] for row in layout_rows]))
                                             for key in CHANNEL_KEYS}}
        if arm == "fft_only" and not fft_independent:
            arm_row["paired_errors_vs_individual"] = {
                key: sum(row["paired_errors_vs_individual"][key] for row in layout_rows)
                for key in layout_rows[0]["paired_errors_vs_individual"]}
        arms.append(arm_row)
        print(f"{arm}: {failures}/{SHOTS}")
    coupling = ("independent Stim seeds" if fft_independent else
                "matched Stim seeds and verified identical detector and observable records for every shot")
    result = {**manifest, "arms": arms, "fft_only_vs_individual_sampling": coupling,
              "interval_scope": "Pooled binomial intervals describe Monte Carlo precision conditional on the five fixed layouts.",
              "score_engine_source_sha256": {name: digest((engine_path / name).read_bytes())
                                              for name in ("engine.py", "engine.cpp", "build_engine.py")}}
    write(HERE / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, nargs="?")
    parser.add_argument("--arms", choices=SCORED_ARMS, nargs="+", default=SCORED_ARMS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--fft-independent", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate(args.fft_independent)
        return
    if args.upstream is None:
        parser.error("An upstream checkout is required for circuit generation")
    upstream, _, information, circuits, info = prepare(args.upstream.resolve())
    write(HERE / "manifest.json", {**info, "layout_metadata_by_arm": information})
    if args.prepare_only:
        print(json.dumps(info, indent=2))
        return
    sys.path.insert(0, str(HERE.parent / "million-engine-03"))
    import engine

    if args.timing_only:
        benchmark(upstream, circuits, information, info, engine)
        return
    pending = []
    for arm in args.arms:
        for index, (circuit_data, metadata) in enumerate(zip(circuits[arm], information[arm], strict=True)):
            paired, fitted_decoder = None, None
            if arm == "fft_only" and not args.fft_independent:
                native, path = native_record("individual", information["individual"][index])
                if native is None or native["shots"] != LAYOUT_SHOTS:
                    pending.append(relative(path))
                    continue
                paired = native["chunks"]
                _, fitted_prior, fitted_flags = circuits["individual"][index]
                fitted_decoder = engine.FastErasureDecoder(upstream, fitted_prior, fitted_flags)
            run_layout(upstream, arm, index, circuit_data, metadata, info["versions"], engine, paired, fitted_decoder)
    if pending:
        raise FileNotFoundError(f"Incomplete native counterparts: {', '.join(pending)}")


if __name__ == "__main__":
    main()
