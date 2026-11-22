import csv
import json

import numpy as np
import pymatching
import pytest
from scipy.stats import binomtest

from bascd.circuits import memory_circuit
from bascd.frontier_audit import audit
from bascd.stats import wilson95


@pytest.fixture
def archive(tmp_path):
    case = tmp_path / "cases" / "tiny-case"
    case.mkdir(parents=True)
    info = dict(case=case.name, basis="z", distance=3, rounds=3, p=.03, variant="ordinary", study="sweep")
    methods = ("mwpm", "candidate")
    timings = {name: [1.0, 2.0, 9.0] for name in methods}
    singles = {name: [1.0, 2.0, 3.0] for name in methods}
    setups = {name: .04 for name in methods}
    seed, shots = 2341, 113
    planned = dict(info, seeds=[seed], shots_per_seed=shots, timing_repeats=3, latency_shots=3)
    manifest = dict(planned, timings=timings, single_us=singles, setup_s=setups)
    (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    protocol = dict(cases=[planned], methods=methods)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    circuit = memory_circuit("z", 3, 3, info["p"])
    circuit.to_file(case / "circuit.stim")
    detectors, truth = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True, bit_packed=True,
    )
    matcher = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    mwpm = matcher.decode_batch(detectors, bit_packed_shots=True, bit_packed_predictions=True)
    np.savez_compressed(
        case / f"seed-{seed}.npz", detectors=detectors, truth=truth,
        mwpm=np.packbits(mwpm[:, 0], bitorder="little"),
        candidate=np.packbits(truth[:, 0], bitorder="little"),
    )
    baseline_errors = np.count_nonzero(mwpm != truth)
    rows = []
    for method, errors, fixed in (("mwpm", baseline_errors, 0), ("candidate", 0, baseline_errors)):
        low, high = wilson95(errors, shots)
        rows.append(dict(
            info, method=method, errors=errors, shots=shots, ler_percent=100 * errors / shots,
            wilson_low=100 * low, wilson_high=100 * high, batch_us_shot=2., batch_us_round=2 / 3,
            single_median_us=2., single_p95_us=2.9, setup_s=.04, fixed=fixed, added=0,
            paired_p=binomtest(fixed, fixed).pvalue if fixed else 1.,
        ))
    with (tmp_path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path, case / f"seed-{seed}.npz"


def test_audit_verifies_archive_and_replays_stim(archive):
    root, _ = archive
    assert audit(root, resample=True) == {
        "cases": 1, "metrics_rows": 2, "seed_archives": 1, "shots": 113,
        "stim_replayed": True, "mwpm_replayed": True,
    }


def test_audit_rejects_padded_prediction_bits(archive):
    root, seed_file = archive
    with np.load(seed_file) as stored:
        data = dict(stored)
    data["candidate"][-1] |= 0x80
    np.savez_compressed(seed_file, **data)
    with pytest.raises(ValueError, match="nonzero candidate padding"):
        audit(root)
