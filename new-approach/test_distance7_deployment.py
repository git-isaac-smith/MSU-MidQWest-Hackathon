import json

import numpy as np
import pytest
import stim

from distance7_study import DeployedDistanceSeven, digest, overridden, protocol_for
from hybrid_decode import main


def write_policy(folder, cutoff):
    plan = protocol_for(None, list_matching=True)
    path = folder / "protocol.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    (folder / "policy.json").write_text(json.dumps(dict(cutoff=cutoff, protocol_sha256=digest(path))),
                                       encoding="utf-8")


@pytest.mark.parametrize("cutoff", [0.5, 1.0])
def test_distance7_deployment_reads_stim_b8_and_matches_policy(tmp_path, cutoff):
    write_policy(tmp_path, cutoff)
    decoder = DeployedDistanceSeven(tmp_path)
    count = decoder.ref.circuit.num_detectors
    rows = decoder.ref.circuit.compile_detector_sampler(seed=93269).sample(1600)
    packed = np.packbits(rows, axis=1, bitorder="little")
    baseline, routed, p_one, _ = decoder.engine.scores(packed)
    assert routed.any() == (cutoff < 1)
    expected = overridden(baseline, routed, p_one, cutoff)
    source, out = tmp_path / "syndromes.b8", tmp_path / "predictions.b8"
    stim.write_shot_data_file(data=rows, path=source, format="b8", num_detectors=count)
    main(["--policy", str(tmp_path), "--mode", "distance7", "--detectors", str(source),
          "--out", str(out), "--batch-size", "113"])
    output = stim.read_shot_data_file(path=out, format="b8", num_observables=1)
    np.testing.assert_array_equal(output[:, 0], expected)
    with pytest.raises(ValueError, match="one column"):
        decoder.predict(rows[:, :-1])
    with pytest.raises(ValueError, match="binary"):
        decoder.predict(rows[:2].astype(np.uint8) + 2)


def test_mismatched_distance7_policy_is_rejected(tmp_path):
    write_policy(tmp_path, 0.5)
    with (tmp_path / "protocol.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="does not match"):
        DeployedDistanceSeven(tmp_path)
