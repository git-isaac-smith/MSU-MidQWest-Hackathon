import json

import numpy as np
import pytest
import stim

from full_map_decoder import FullMapDecoder
from hybrid_decode import main


def test_detector_only_full_map_b8_needs_only_physical_policy(tmp_path):
    (tmp_path / "policy.json").write_text(json.dumps({"physical": [3, 0.008, 0.04], "rounds": 5}))
    decoder = FullMapDecoder.load(tmp_path, "reference")
    circuit = decoder.ref.circuit
    detectors = circuit.compile_detector_sampler(seed=93268).sample(251)
    source, output = tmp_path / "detectors.b8", tmp_path / "predictions.b8"
    stim.write_shot_data_file(data=detectors, path=source, format="b8", num_detectors=circuit.num_detectors)
    main(["--mode", "full_map", "--policy", str(tmp_path), "--detectors", str(source),
          "--out", str(output), "--batch-size", "31"])
    decoded = stim.read_shot_data_file(path=output, format="b8", num_observables=1)
    np.testing.assert_array_equal(decoded[:, 0], decoder.predict(detectors))


def test_numerical_ties_and_incompatible_rounds(tmp_path):
    decoder = FullMapDecoder([3, 0.008, 0.04], "reference")
    d = decoder.ref.circuit.compile_detector_sampler(seed=17391).sample(5)
    p = np.array([0.5, 0.5 - 1e-13, 0.5 + 1e-13, 0.8, 0.2])
    expected = decoder.ref.mwpm_parity(d)
    expected[-2:] = [True, False]
    np.testing.assert_array_equal(decoder.decisions(d, p), expected)
    (tmp_path / "policy.json").write_text(json.dumps({"physical": [3, 0.008, 0.04], "rounds": 7}))
    with pytest.raises(ValueError, match="five"):
        FullMapDecoder.load(tmp_path)
