"""Temporal diagnostics for packed Stim rotated-memory detector events."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import stim


def summarize(
    case_dir: Path, circuit: stim.Circuit, records: np.ndarray,
) -> dict[str, Path]:
    """Save detector means and temporal periodograms from independent packed shots.

    Each mean-subtracted trace contains one check coordinate across consecutive
    syndrome rounds. The terminal data-readout closure is kept in the detector
    table and excluded from the periodograms because it measures a different
    combination of record bits. Checks with equal time ranges share a frequency
    grid. Power is averaged over checks in each range.
    """
    coordinates = circuit.get_detector_coordinates()
    if records.dtype != np.uint8 or records.ndim != 2:
        raise ValueError("expected a two-dimensional array of uint8 packed detector records")
    if records.shape[1] != (circuit.num_detectors + 7) // 8 or not len(records):
        raise ValueError("packed detector records must contain shots matching the circuit")
    if not coordinates or any(len(coord) != 3 for coord in coordinates.values()):
        raise ValueError("expected three-dimensional detector coordinates")

    closing_time = max(coord[2] for coord in coordinates.values())
    checks = defaultdict(list)
    for detector, (x, y, time) in coordinates.items():
        if time < closing_time:
            checks[x, y].append((time, detector))

    groups = defaultdict(list)
    for check in checks.values():
        check.sort()
        times, detector_ids = zip(*check)
        if times != tuple(range(int(times[0]), int(times[-1]) + 1)):
            raise ValueError("expected contiguous, unit-spaced syndrome rounds")
        if len(times) > 1:
            groups[times].append(detector_ids)
    groups = {times: np.array(ids) for times, ids in sorted(groups.items())}

    counts = np.zeros(circuit.num_detectors, dtype=np.int64)
    shot_power_sum = {times: np.zeros(len(times) // 2 + 1) for times in groups}
    for start in range(0, len(records), 4096):
        shots = np.unpackbits(
            records[start:start + 4096], axis=1, count=circuit.num_detectors, bitorder="little",
        )
        counts += shots.sum(axis=0, dtype=np.int64)
        for times, ids in groups.items():
            traces = shots[:, ids].astype(np.float64)
            traces -= traces.mean(axis=-1, keepdims=True)
            shot_power_sum[times] += np.square(np.abs(np.fft.rfft(traces))).sum(axis=(0, 1))

    means = counts / len(records)
    case_dir.mkdir(parents=True, exist_ok=True)
    detector_path = case_dir / "detector-time-mean.csv"
    with detector_path.open("w", newline="", encoding="utf-8") as file:
        rows = csv.writer(file)
        rows.writerow(("detector", "x", "y", "round", "data_readout_closure", "events", "shots",
                       "event_probability"))
        for detector, (x, y, time) in coordinates.items():
            rows.writerow((detector, x, y, time, int(time == closing_time), counts[detector],
                           len(records), means[detector]))

    power_path = case_dir / "temporal-power.csv"
    with power_path.open("w", newline="", encoding="utf-8") as file:
        rows = csv.writer(file)
        rows.writerow(("first_round", "last_round", "trace_length", "checks", "shots",
                       "frequency_cycles_per_round", "mean_shot_power", "mean_trace_power",
                       "shot_noise_in_mean_estimate"))
        for times, ids in groups.items():
            length = len(times)
            # The one-sided spectrum sums to the mean squared centered trace.
            scale = np.full(length // 2 + 1, 2 / length**2)
            scale[0] /= 2
            if length % 2 == 0:
                scale[-1] /= 2
            single = shot_power_sum[times] * scale / (len(records) * len(ids))
            mean_traces = means[ids]
            mean_traces -= mean_traces.mean(axis=-1, keepdims=True)
            mean = np.square(np.abs(np.fft.rfft(mean_traces))).mean(axis=0) * scale
            for i, frequency in enumerate(np.fft.rfftfreq(length)):
                noise = max(0.0, single[i] - mean[i]) / (len(records) - 1) if len(records) > 1 else ""
                rows.writerow((times[0], times[-1], length, len(ids), len(records), frequency,
                               single[i], mean[i], noise))

    return {"detector_means": detector_path, "temporal_power": power_path}
