import json
from types import SimpleNamespace

import numpy as np
import pytest
import stim

from frequency import Passband
from hybrid_cnn import HybridCNN
from hybrid_decode import decode_file, main
from hybrid_policy import HybridDecoder
from localizer import Localizer


class DummyDecoder:
    def __init__(self, detectors=10, observables=1):
        self.ref = SimpleNamespace(circuit=SimpleNamespace(num_detectors=detectors, num_observables=observables))
        self.batches = []

    def predict(self, detectors, mode):
        assert detectors.dtype == np.bool_
        self.batches.append((mode, detectors.copy()))
        return detectors[:, 0] ^ detectors[:, -1]


def test_b8_round_trip_matches_stim_and_respects_batch_size(tmp_path):
    decoder = DummyDecoder()
    source, output = tmp_path / "detectors.b8", tmp_path / "output.b8"
    shots = np.zeros((5, 10), dtype=bool)
    shots[[0, 2, 3], 0] = True
    shots[[1, 2], 9] = True
    shots[4, 7] = True
    stim.write_shot_data_file(path=source, format="b8", num_detectors=10, data=shots)

    assert decode_file(decoder, source, output, mode="learned", batch_size=2) == 5
    assert [mode for mode, _ in decoder.batches] == ["learned"] * 3
    assert [len(batch) for _, batch in decoder.batches] == [2, 2, 1]
    np.testing.assert_array_equal(np.concatenate([batch for _, batch in decoder.batches]), shots)
    assert output.read_bytes() == b"\x01\x01\x00\x01\x00"
    predictions = stim.read_shot_data_file(path=output, format="b8", num_observables=1)
    np.testing.assert_array_equal(predictions[:, 0], shots[:, 0] ^ shots[:, -1])


def test_empty_b8_writes_empty_predictions_without_calling_decoder(tmp_path):
    decoder = DummyDecoder()
    source, output = tmp_path / "empty.b8", tmp_path / "output.b8"
    source.touch()
    assert decode_file(decoder, source, output) == 0
    assert not decoder.batches
    assert output.read_bytes() == b""


@pytest.mark.parametrize(
    ("bad_records", "message"),
    [(b"\x00\x00\x01", "Partial b8 detector record"), (b"\x00\x00\x01\x04", "Nonzero padding bits")],
)
def test_invalid_input_leaves_no_output(tmp_path, bad_records, message):
    decoder = DummyDecoder()
    source, output = tmp_path / "invalid.b8", tmp_path / "output.b8"
    source.write_bytes(bad_records)
    with pytest.raises(ValueError, match=message):
        decode_file(decoder, source, output, batch_size=1)
    assert source.read_bytes() == bad_records
    assert not output.exists()


@pytest.mark.parametrize("dimensions", [(0, 1), (10, 0), (10, 2)])
def test_policy_circuit_dimensions_are_checked(tmp_path, dimensions):
    decoder = DummyDecoder(*dimensions)
    source, output = tmp_path / "detectors.b8", tmp_path / "output.b8"
    source.touch()
    with pytest.raises(ValueError, match="at least one detector and exactly one observable"):
        decode_file(decoder, source, output)
    assert not output.exists()


def test_predictions_require_one_binary_value_per_shot(tmp_path):
    decoder = DummyDecoder()
    source, output = tmp_path / "detectors.b8", tmp_path / "output.b8"
    source.write_bytes(b"\x00\x00")
    for prediction in (np.zeros((1, 1), dtype=bool), np.array([2], dtype=np.uint8)):
        decoder.predict = lambda *args, **kwargs: prediction
        with pytest.raises(ValueError, match="one binary logical prediction per shot"):
            decode_file(decoder, source, output)
        assert not output.exists()


def test_never_overwrites_existing_output_or_input(tmp_path):
    decoder = DummyDecoder()
    source, output = tmp_path / "detectors.b8", tmp_path / "output.b8"
    source.write_bytes(b"\x00\x01")
    output.write_bytes(b"Existing user file")
    for destination in output, source:
        with pytest.raises(FileExistsError, match="Output already exists"):
            decode_file(decoder, source, destination)
    assert source.read_bytes() == b"\x00\x01"
    assert output.read_bytes() == b"Existing user file"
    assert not decoder.batches


def test_cli_loads_lock_and_decodes_public_stim_shots(tmp_path, capsys):
    physical = (3, 0.02, 0.04)
    band = Passband()
    site = Localizer(7, seed=12)
    decoder = HybridDecoder(physical, site, 0.5, band)
    specs = {
        "physical": physical,
        "passband": {"temporal": band.temporal, "spatial": band.spatial},
        "site_threshold": 0.5,
        "gap_limit": 100.0,
        "override_threshold": 0.5,
        "cnn_route_at": -0.2,
        "gap_route_at": -1.5,
    }
    source, output = tmp_path / "detectors.b8", tmp_path / "output.b8"
    circuit = decoder.ref.circuit
    circuit.compile_detector_sampler(seed=13).sample_write(9, filepath=source, format="b8")
    shots = stim.read_shot_data_file(path=source, format="b8", num_detectors=circuit.num_detectors)
    baseline = decoder.ref.mwpm_parity(shots)
    features = decoder.encode(shots, baseline, decoder.confidence.features(shots))
    decoder.network = HybridCNN(features, seed=14, conv_channels=4, hidden_channels=8)
    decoder.policy = specs
    site.save(tmp_path / "site-cnn.npz")
    decoder.network.save(tmp_path / "logical-cnn.npz")
    (tmp_path / "policy.json").write_text(json.dumps(specs), encoding="utf-8")

    main([
        "--policy", str(tmp_path), "--detectors", str(source), "--out", str(output),
        "--mode", "accuracy", "--batch-size", "4",
    ])
    assert "Decoded 9 shots in accuracy mode" in capsys.readouterr().out
    expected = decoder.predict(shots, mode="accuracy")
    predictions = stim.read_shot_data_file(path=output, format="b8", num_observables=1)
    np.testing.assert_array_equal(predictions[:, 0], expected)
