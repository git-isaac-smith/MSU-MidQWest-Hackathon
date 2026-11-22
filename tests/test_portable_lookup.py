import json

import numpy as np
import pytest
import stim

from bascd.exact_small import compile_exact_small
from bascd.portable_lookup import PortableLookup, _compile, _export, predict_file


def test_table_uses_raw_little_endian_indices_and_applies_both_offsets():
    model = stim.DetectorErrorModel("""
        error(1) D0 D8 L0
        error(0.1) D0 L0
        error(0.2) D8
        logical_observable L0
    """)
    exact = compile_exact_small(model)
    lookup = PortableLookup.from_exact(exact)
    indices = np.arange(512, dtype="<u2")
    rows = indices.view(np.uint8).reshape(-1, 2)
    reference = exact.decode_batch(np.unpackbits(rows, axis=1, count=9, bitorder="little"))
    np.testing.assert_array_equal(lookup.predict_batch(rows), reference)
    assert [lookup.predict(row.tobytes()) for row in rows] == reference[:, 0].tolist()
    assert lookup.predict(bytes([1, 1])) == 1
    assert lookup.predict(bytes([0, 1])) == 0
    assert lookup.predict_batch(rows[:0]).shape == (0, 1)
    with pytest.raises(ValueError, match="padding"):
        lookup.predict(bytes([0, 2]))
    with pytest.raises(ValueError, match="padding"):
        lookup.predict_batch(np.array([[0, 2]], dtype=np.uint8))


def test_export_verified_and_file_prediction_protects_output(tmp_path):
    circuit_path = tmp_path / "source.stim"
    circuit_path.write_text("R 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\n"
                            "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n", encoding="utf-8")
    source, _, exact, compiled, _, _ = _compile(circuit_path)
    export = tmp_path / "export"
    _export(export, source, exact, compiled)
    lookup = PortableLookup.load(export)
    assert lookup == compiled
    with pytest.raises(FileExistsError):
        _export(export, source, exact, compiled)
    incoming, outgoing = tmp_path / "in.b8", tmp_path / "out.b8"
    incoming.write_bytes(bytes([0, 1]))
    predict_file(lookup, incoming, outgoing)
    assert outgoing.read_bytes() == bytes([0, 1])
    with pytest.raises(FileExistsError):
        predict_file(lookup, incoming, outgoing)
    invalid_output = tmp_path / "invalid.b8"
    incoming.write_bytes(bytes([4]))
    with pytest.raises(ValueError, match="padding"):
        predict_file(lookup, incoming, invalid_output)
    assert not invalid_output.exists()

    manifest_path = export / "manifest.json"
    original_metadata = manifest_path.read_text(encoding="utf-8")
    metadata = json.loads(original_metadata)
    metadata["record_bytes"] = 2
    manifest_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        PortableLookup.load(export)
    manifest_path.write_text(original_metadata, encoding="utf-8")
    (export / "decisions.b8").write_bytes(bytes([0]))
    with pytest.raises(ValueError, match="table"):
        PortableLookup.load(export)


def test_partial_input_record_leaves_no_output(tmp_path):
    lookup = PortableLookup(9, bytes(64))
    incoming, outgoing = tmp_path / "in.b8", tmp_path / "out.b8"
    incoming.write_bytes(bytes([0, 1, 2]))
    with pytest.raises(ValueError, match="partial"):
        predict_file(lookup, incoming, outgoing)
    assert not outgoing.exists()
