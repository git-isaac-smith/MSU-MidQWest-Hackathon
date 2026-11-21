import json

import numpy as np
import pytest
import stim

from bascd.accuracy_decoders import circuit_hash
from bascd.decode_accuracy import decode_file


@pytest.mark.parametrize("deduplicate", [False, True])
def test_b8_decodes_partial_byte_and_refuses_incomplete_shots(tmp_path, deduplicate):
    circuit = stim.Circuit("R 0\nX_ERROR(.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    circuit += stim.Circuit("DETECTOR\n") * 8
    circuit_file, policy, events, output = [tmp_path / x for x in ("c.stim", "policy.json", "in.b8", "out.b8")]
    circuit.to_file(circuit_file)
    policy.write_text(json.dumps(dict(format_version=1, by_distance={"3":"exact_small"}, recipe={},
                                      circuits=[dict(sha256=circuit_hash(circuit), scenario={"distance":3})])))
    events.write_bytes(bytes([1,0, 0,0, 1,0]))
    assert decode_file(policy, circuit_file, events, output, 3, deduplicate=deduplicate) == 3
    result = stim.read_shot_data_file(path=output, format="b8", num_observables=1)
    np.testing.assert_array_equal(result, [[True], [False], [True]])
    with pytest.raises(FileExistsError):
        decode_file(policy, circuit_file, events, output)
    events.write_bytes(bytes([1,0,1]))
    with pytest.raises(ValueError, match="incomplete"):
        decode_file(policy, circuit_file, events, tmp_path/"bad.b8", 1)
    assert not (tmp_path/"bad.b8").exists()
