import csv
import json

import numpy as np
import pytest
import stim

from bascd.accuracy_decoders import circuit_hash
from bascd.circuits import memory_circuit
from bascd.measure import Scenario
from bascd.stim_samples import run


def test_archive_replays_separate_samples_and_recounts_failures(tmp_path):
    pytest.importorskip("stabilizer_qec")
    physical = memory_circuit("x", 3, 3, .01)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "format_version": 1, "by_distance": {"3": "native"},
        "recipe": {"bath_xyz": {"clock": [.4, .3, .3]}},
        "circuits": [{"sha256": circuit_hash(physical),
                      "scenario": {"basis": "x", "distance": 3, "rounds": 3, "p": .01}}],
    }))
    case = Scenario(3, .01, "x")
    archive = tmp_path / "archive"
    summary = run(archive, policy, [case], 513, 179, ["smoke"])
    folder = next((archive / "cases").iterdir())
    original = stim.Circuit.from_file(folder / "circuit.stim")
    expected, truth = original.compile_detector_sampler(seed=case.seed(179, "final_test")).sample(
        513, separate_observables=True, bit_packed=True)
    assert (folder / "detectors.b8").read_bytes() == expected.tobytes()
    assert (folder / "observables.b8").read_bytes() == truth.tobytes()
    with (archive / "stats.csv").open(newline="") as src:
        for row in csv.DictReader(src):
            pred = stim.read_shot_data_file(path=folder/(row["decoder"] + ".b8"), format="b8",
                                            num_observables=1, bit_packed=True)
            assert int(row["failures"]) == np.any(pred != truth, axis=1).sum()
    assert summary["paired"]["native"]["recovered"] == summary["paired"]["native"]["lost"] == 0
    assert (archive / "frozen-policy.json").read_bytes() == policy.read_bytes()
    with pytest.raises(FileExistsError):
        run(archive, policy, [case], 513, 179, ["smoke"])
