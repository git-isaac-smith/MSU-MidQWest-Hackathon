"""Time the author's warmed decoder on exact prefixes of the scored batches."""

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from fft_replay import fft_channels
from run import BASIS, DISTANCE, HERE, channels_for_run, load_pinned, replay, write_json


PREFIX_SHOTS = 512
WARM_CALLS = 1
TIMED_CALLS = 3
ORDER_SEED = 6862176547


@dataclass
class Batch:
    arm: str
    layout: int
    decoder: object
    detectors: np.ndarray
    provenance: dict
    times_ns: list = field(default_factory=list)
    warm_prediction: np.ndarray | None = None


def make_batch(upstream, arm, index, layout, pairs, archive):
    sites = zip(layout["bath_ids"], layout["times"][arm], strict=True)
    physical, priors = zip(*(pairs[site] for site in sites), strict=True)
    circuit, flags = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(physical), upstream.P_MEAS)
    prior, prior_flags = upstream.build_circuit(DISTANCE, BASIS, *upstream._rates(priors), upstream.P_MEAS)
    for key, candidate in (("physical_circuit_sha256", circuit), ("prior_circuit_sha256", prior)):
        if sha256(str(candidate).encode()).hexdigest() != archive[key]:
            raise ValueError(f"{arm} layout {index} {key} differs from scored archive")
    if not np.array_equal(flags, prior_flags):
        raise ValueError(f"{arm} layout {index} mismatch in herald structure")

    seed, shots = archive["stim_seed"], archive["shots"]
    detectors, labels = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    archive_digest = sha256(np.packbits(detectors).tobytes() + np.packbits(labels).tobytes()).hexdigest()
    if shots < PREFIX_SHOTS or archive_digest != archive["sample_sha256"]:
        raise ValueError(f"{arm} layout {index} full sample differs from scored archive")
    prefix = detectors[:PREFIX_SHOTS].copy()
    provenance = dict(stim_seed=seed, source_shots=shots, source_sample_sha256=archive_digest,
                      prefix_detector_sha256=sha256(np.packbits(prefix).tobytes()).hexdigest())
    decoder = upstream.ErasureDecoder(prior, flags)
    if decoder.skipped:
        raise ValueError(f"{arm} layout {index} decoder skipped non-graphlike edges")
    return Batch(arm, index, decoder, prefix, provenance)


def load_batches(upstream, params_path, calibration, main, fft_calibration, fft):
    layouts = calibration["frozen_layouts"]
    for layout in layouts:
        for arm, times in fft_calibration["times_by_arm_and_bath"].items():
            layout["times"][arm] = [times[str(bath)] for bath in layout["bath_ids"]]
    params = {int(row["Bath"]): row for row in replay.read_rows(params_path) if row["Set"] == "unseen"}
    batches = []
    for archived_arm in [*main["arms"], *fft["arms"]]:
        arm = archived_arm["arm"]
        if arm == "fft_only":
            times = {int(bath): tau for bath, tau in fft_calibration["times_by_arm_and_bath"][arm].items()}
            inputs = {int(bath): row for bath, row in fft_calibration["fft_frame_and_prior_by_bath"].items()}
            pairs, _ = fft_channels(upstream, times, inputs, params)
        else:
            pairs, _ = channels_for_run(upstream, arm, layouts, params, None)
        batches.extend(make_batch(upstream, arm, index, layout, pairs, row)
                       for index, (layout, row) in enumerate(zip(layouts, archived_arm["layouts"], strict=True), 1))
    return batches


def time_batches(batches):
    for batch in batches:
        for _ in range(WARM_CALLS):
            batch.warm_prediction = batch.decoder.decode(batch.detectors)

    rng = np.random.default_rng(ORDER_SEED)
    order = [rng.permutation(len(batches)).tolist() for _ in range(TIMED_CALLS)]
    for indices in order:
        for index in indices:
            batch = batches[index]
            start_ns = perf_counter_ns()
            predicted = batch.decoder.decode(batch.detectors)
            elapsed_ns = perf_counter_ns() - start_ns
            if not np.array_equal(batch.warm_prediction, predicted):
                raise ValueError(f"{batch.arm} layout {batch.layout} warm and timed predictions differ")
            batch.times_ns.append(elapsed_ns)
    return [[dict(arm=batches[i].arm, layout=batches[i].layout) for i in indices] for indices in order]


def summarize(batches):
    grouped = {}
    for batch in batches:
        grouped.setdefault(batch.arm, []).append(dict(layout=batch.layout, **batch.provenance,
                                                      batch_times_ns=batch.times_ns,
                                                      median_batch_ns=int(np.median(batch.times_ns))))
    arms = []
    for arm, layouts in grouped.items():
        us_per_shot = sum(row["median_batch_ns"] for row in layouts) / (len(layouts) * PREFIX_SHOTS * 1_000)
        arms.append(dict(arm=arm, batch_us_shot=us_per_shot, batch_us_round=us_per_shot / DISTANCE,
                         layouts=layouts))
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--archive", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "timing.json")
    args = parser.parse_args()
    upstream, rev, params_path, _ = load_pinned(args.upstream.resolve())
    names = ("calibration.json", "result.json", "fft_calibration.json", "fft_result.json")
    blobs = {name: (args.archive / name).read_bytes() for name in names}
    provenance = {name: sha256(blob).hexdigest() for name, blob in blobs.items()}
    calibration, main_result, fft_calibration, fft_result = (json.loads(blobs[name]) for name in names)
    if (calibration["upstream_commit"] != rev or fft_calibration["upstream_commit"] != rev
            or main_result["calibration_sha256"] != provenance[names[0]]
            or fft_calibration["control_calibration_sha256"] != provenance[names[0]]
            or fft_result["fft_calibration_sha256"] != provenance[names[2]]
            or calibration["parameter_csv_sha256"] != sha256(params_path.read_bytes()).hexdigest()):
        raise ValueError("Accuracy archive provenance does not reconcile")
    batches = load_batches(upstream, params_path, calibration, main_result, fft_calibration, fft_result)
    if len(batches) != 40:
        raise ValueError("Expected eight arms and five batches per arm")
    print("Verified 40 full seeded samples and constructed decoder graphs", flush=True)
    order = time_batches(batches)
    output = dict(upstream_commit=rev, source_sha256=provenance, platform=platform.platform(),
                  processor=platform.processor(), logical_cpu_count=os.cpu_count(), python=platform.python_version(),
                  archive_versions=main_result["versions"],
                  thread_env={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                  timer="perf_counter_ns around ErasureDecoder.decode only", serial_cpu=True,
                  prefix_shots_per_layout=PREFIX_SHOTS, warm_calls_per_layout=WARM_CALLS,
                  timed_calls_per_layout=TIMED_CALLS, rounds=DISTANCE,
                  aggregation="Mean of the five layout medians divided by prefix shots, then by rounds for per-round cost",
                  order_seed=ORDER_SEED, timed_call_order=order, arms=summarize(batches))
    write_json(args.output, output)
    for arm in output["arms"]:
        print(f"{arm['arm']}: {arm['batch_us_shot']:.1f} us/shot, {arm['batch_us_round']:.1f} us/round")


if __name__ == "__main__":
    main()
