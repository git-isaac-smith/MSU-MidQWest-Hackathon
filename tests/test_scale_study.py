import csv

import numpy as np
import pytest
import stim

from bascd import scale_study
from bascd.scale_methods import _find_root, _growth_and_peel, _handwritten_graph


def test_prepare_records_replay_the_fixed_circuit(tmp_path, monkeypatch):
    monkeypatch.setattr(scale_study, "DISTANCES", [3])
    monkeypatch.setattr(scale_study, "SHOTS", 9)
    monkeypatch.setattr(scale_study, "version", lambda package: "test")
    scale_study.prepare(tmp_path)

    protocol = scale_study.load(tmp_path / "protocol.json")
    assert protocol["p"] == 0.005
    assert {meta["basis"] for meta in protocol["cases"]} == {"x", "z"}
    all_seeds = set()
    for meta in protocol["cases"]:
        assert meta["distance"] == meta["rounds"] == 3
        assert (meta["data_qubits"], meta["physical_qubits"]) == (9, 17)
        all_seeds.update((meta["seed"], meta["training_seed"]))
        folder = tmp_path / meta["folder"]
        circuit = stim.Circuit.from_file(folder / "circuit.stim")
        assert scale_study.digest(str(circuit).encode()) == meta["circuit_sha256"]
        detectors, truth = circuit.compile_detector_sampler(seed=meta["seed"]).sample(
            9, separate_observables=True, bit_packed=True,
        )
        with np.load(folder / "records.npz") as archive:
            np.testing.assert_array_equal(archive["detectors"], detectors)
            np.testing.assert_array_equal(archive["truth"], truth)
        assert scale_study.digest(detectors.tobytes() + truth.tobytes()) == meta["sample_sha256"]
    assert len(all_seeds) == 4


def test_worker_and_collect_score_only_the_paired_prefix(tmp_path, monkeypatch):
    total, prefix = 131, 129
    monkeypatch.setattr(scale_study, "SHOTS", total)
    monkeypatch.setattr(scale_study, "planned_shots", lambda name, distance: total if name == "mwpm" else prefix)
    case = tmp_path / "d3-z"
    case.mkdir()
    circuit = scale_study.memory_circuit("z", 3, 3, 0.005)
    circuit.to_file(case / "circuit.stim")
    meta = dict(distance=3, basis="z", p=0.005, data_qubits=9, physical_qubits=17, rounds=3,
                folder=case.name, training_seed=200,
                detectors=circuit.num_detectors, seed=201)
    records = np.zeros((total, circuit.num_detectors), dtype=bool)
    records[[0, 1, 129, 130], -1] = True
    records[[1, 2, 128], 0] = True
    packed = np.packbits(records, axis=1, bitorder="little")
    truth = np.zeros((total, 1), dtype=np.uint8)
    truth[[1, 2, 128], 0] = 1
    meta.update(circuit_sha256=scale_study.digest(str(circuit).encode()),
                sample_sha256=scale_study.digest(packed.tobytes() + truth.tobytes()))
    np.savez_compressed(case / "records.npz", detectors=packed, truth=truth)
    scale_study.save(case / "case.json", meta)
    scale_study.save(tmp_path / "protocol.json", {"cases": [meta]})

    observed_batches = {}

    def fake_factory(name, circuit, meta, folder):
        column = -1 if name == "mwpm" else 0
        observed_batches[name] = []

        def decode(bits):
            assert bits.dtype == np.bool_
            assert bits.shape[1] == circuit.num_detectors
            observed_batches[name].append(len(bits))
            result = bits[:, column, None].copy()
            if name == "correlated" and len(bits) == 128:
                result[3] = True
            return result

        return decode

    monkeypatch.setattr(scale_study, "make_decoder", fake_factory)
    scale_study.worker(case, ["mwpm", "correlated"])
    assert observed_batches == {"mwpm": [3, 128, 3], "correlated": [3, 128, 1]}
    assert scale_study.load(case / "correlated.json")["status"] == "pilot"
    assert len(scale_study.load(case / "correlated.json")["batch_ns"]) == 1
    scale_study.collect(tmp_path)
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as file:
        rows = {row["method"]: row for row in csv.DictReader(file)}
    assert (rows["mwpm"]["shots"], rows["mwpm"]["failures"]) == ("131", "5")
    assert (rows["correlated"]["shots"], rows["correlated"]["failures"]) == ("129", "1")
    assert (rows["correlated"]["recovered"], rows["correlated"]["lost"]) == ("3", "1")

    tampered = np.load(case / "correlated.npy", allow_pickle=False)
    tampered[0] ^= 1
    np.save(case / "correlated.npy", tampered, allow_pickle=False)
    with pytest.raises(ValueError, match="Invalid predictions"):
        scale_study.collect(tmp_path)


def test_handwritten_union_find_peels_first_fault_and_boundary():
    model = stim.DetectorErrorModel("error(0.01) D0 D1\nerror(0.02) D1 L0")
    detectors = np.array([[False, False], [True, False], [False, True], [True, True]])
    predictions = _growth_and_peel(detectors, *_handwritten_graph(model), _find_root)
    np.testing.assert_array_equal(predictions, [[0], [1], [1], [0]])
