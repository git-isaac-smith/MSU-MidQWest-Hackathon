"""Prepare ambiguous-syndrome splits for learning ordinary MWPM's error risk."""

from math import isfinite
from operator import index
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching
import stim

from logical_reference import build_reference
from matching_confidence import MatchingConfidence
from posterior_decoder import PosteriorDecoder


SAMPLE_BATCH = 8192
TEACHER_BATCH = 2048
TRAIN_THIN_ABOVE = 3.0
TRAIN_RETAIN_FRACTION = 0.25
_REPORT_SECONDS = 20


def prepare_split(path, physical, shots, seed, gap_limit=5.3, train=False):
    """Save a seeded Stim split and exact conditional error risk for retained rows.

    Selection uses the difference between the two minimum matching costs.
    Wider-gap training rows are thinned by an independent seeded RNG. The
    teacher conditions on public checks and estimates the probability that
    ordinary MWPM's decision is wrong. Sampled logical truth is saved for
    evaluation. The scalar `shots` covers the entire sampled split.
    """
    path = Path(path)
    shots, seed = index(shots), index(seed)
    if shots < 0:
        raise ValueError("shots must be nonnegative")
    if not isfinite(gap_limit) or gap_limit < 0:
        raise ValueError("gap_limit must be finite and nonnegative")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    distance, p_data, p_readout = physical
    reference = build_reference(distance, p_data, p_readout)
    matching = MatchingConfidence(distance, p_data, p_readout)
    teacher_decoder = PosteriorDecoder(distance, p_data, p_readout)
    sampler = reference.circuit.compile_detector_sampler(seed=seed)
    subsample_seed = np.random.SeedSequence([seed, 1]).generate_state(1, dtype=np.uint64)[0]
    subsampler = np.random.default_rng(subsample_seed)

    detector_blocks = [np.empty((0, reference.circuit.num_detectors), dtype=bool)]
    cost_blocks = [np.empty((0, 2), dtype=np.float64)]
    baseline_blocks, truth_blocks = [np.empty(0, dtype=bool)], [np.empty(0, dtype=bool)]
    baseline_errors = eligible = retained = 0
    print(f"{path.name}: Sampling {shots:,} Stim shots with seed {seed}", flush=True)
    next_report = perf_counter() + _REPORT_SECONDS
    for start in range(0, shots, SAMPLE_BATCH):
        stop = min(start + SAMPLE_BATCH, shots)
        detectors, logical = sampler.sample(stop - start, separate_observables=True)
        truth = logical[:, 0]
        baseline = reference.mwpm_parity(detectors)
        baseline_errors += np.count_nonzero(baseline ^ truth)
        costs = matching.features(detectors)
        gap = np.full(len(costs), np.inf)
        finite = np.isfinite(costs).all(axis=1)
        gap[finite] = np.abs(costs[finite, 1] - costs[finite, 0])
        keep = gap <= gap_limit
        eligible += np.count_nonzero(keep)
        if train:
            draw = subsampler.random(len(detectors))
            keep &= (gap <= TRAIN_THIN_ABOVE) | (draw < TRAIN_RETAIN_FRACTION)
        retained += np.count_nonzero(keep)
        detector_blocks.append(detectors[keep])
        cost_blocks.append(costs[keep])
        baseline_blocks.append(baseline[keep])
        truth_blocks.append(truth[keep])
        if perf_counter() >= next_report:
            print(f"{path.name}: Sampled {stop:,}/{shots:,} and retained {retained:,}", flush=True)
            next_report = perf_counter() + _REPORT_SECONDS

    detectors = np.concatenate(detector_blocks)
    costs = np.concatenate(cost_blocks)
    baseline = np.concatenate(baseline_blocks)
    truth = np.concatenate(truth_blocks)
    teacher = np.empty(retained, dtype=np.float32)
    print(f"{path.name}: Evaluating posterior risk for {retained:,} of {eligible:,} eligible shots", flush=True)
    next_report = perf_counter() + _REPORT_SECONDS
    for start in range(0, retained, TEACHER_BATCH):
        stop = min(start + TEACHER_BATCH, retained)
        part = slice(start, stop)
        observed, terminal = reference.public_checks(detectors[part])
        probability_one = teacher_decoder.predict_proba(observed, terminal)
        teacher[part] = np.where(baseline[part], 1 - probability_one, probability_one)
        if perf_counter() >= next_report:
            print(f"{path.name}: Evaluated {stop:,}/{retained:,} posterior risks", flush=True)
            next_report = perf_counter() + _REPORT_SECONDS

    with path.open("xb") as archive:
        np.savez_compressed(
            archive,
            detectors=detectors,
            costs=costs,
            baseline=baseline,
            teacher=teacher,
            truth=truth,
            shots=shots,
            baseline_errors=baseline_errors,
            eligible=eligible,
            retained=retained,
            retained_baseline_errors=np.count_nonzero(baseline ^ truth),
            distance=distance,
            p_data=p_data,
            p_readout=p_readout,
            seed=np.uint64(seed),
            subsample_seed=subsample_seed,
            gap_limit=gap_limit,
            train=train,
            train_thin_above=TRAIN_THIN_ABOVE,
            train_retain_fraction=TRAIN_RETAIN_FRACTION,
            sample_batch=SAMPLE_BATCH,
            teacher_batch=TEACHER_BATCH,
            numpy_version=np.__version__,
            pymatching_version=pymatching.__version__,
            stim_version=stim.__version__,
        )
    print(f"{path.name}: Saved {retained:,} rows with {baseline_errors:,} MWPM errors across {shots:,} shots", flush=True)
