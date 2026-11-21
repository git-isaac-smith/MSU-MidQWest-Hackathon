import json

import numpy as np
import pytest
import stim

from bascd.bath_prior import BathPrior
from bascd.circuits import memory_circuit
from bascd.ensemble import (
    EnsembleDecoder,
    EnsembleMember,
    EnsembleSpec,
    PreparedEnsembleCase,
    jitter_model,
)


def test_jitter_preserves_complete_errors_and_detector_coordinates():
    model = stim.DetectorErrorModel("""
        error[first](0.05) D0 D1 ^ D2 L0
        detector(1, 2, 0) D0
        repeat 2 {
            error[repeated](0.03) D1 ^ D2
            detector(1, 2, 1) D1
            shift_detectors(0, 0, 1) 1
        }
        error(0) D4
        error(0.5) D4
    """)
    jittered = jitter_model(model, 0.5, 29)
    assert jittered == jitter_model(model, 0.5, 29)
    assert jittered != jitter_model(model, 0.5, 47)
    assert jittered.get_detector_coordinates() == model.get_detector_coordinates()
    assert jittered.num_observables == model.num_observables
    for before, after in zip(model.flattened(), jittered, strict=True):
        assert before.type == after.type
        assert before.tag == after.tag
        assert before.targets_copy() == after.targets_copy()
        if before.type != "error" or before.args_copy()[0] in (0, 0.5):
            assert before.args_copy() == after.args_copy()
        else:
            assert before.args_copy() != after.args_copy()
    repeated = [item.args_copy()[0] for item in jittered if item.tag == "repeated"]
    assert len(set(repeated)) == 2


def test_compiled_ensemble_matches_cached_members():
    circuit = memory_circuit("z", 3, 3, 0.006)
    model = circuit.detector_error_model(decompose_errors=True)
    clock = BathPrior("clock", (0.8, 0.05, 0.15))
    case = PreparedEnsembleCase(circuit, model, basis="z", distance=3, rounds=3, p=0.006,
                                priors={"clock": clock})
    spec = EnsembleSpec((EnsembleMember(), EnsembleMember("clock"),
                         EnsembleMember(jitter=0.3, seed=11),
                         EnsembleMember(jitter=0.3, seed=29),
                         EnsembleMember(jitter=0.3, seed=47)))
    assert EnsembleSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    detectors = circuit.compile_detector_sampler(seed=307).sample(500)
    detectors.flags.writeable = False
    predictions = case.predict_members(detectors, (*spec.members, spec.members[0]))
    deltas = [prediction[:, 0] != predictions[spec.members[0]][:, 0]
              for prediction in predictions.values()]
    expected_disagreement = np.any(deltas, axis=0)
    assert len(predictions) == len(spec.members)
    assert expected_disagreement.any()
    decoder = case.compile(spec)
    expected = spec.vote(predictions)
    batch, disagreement = decoder.predictions_and_disagreement(detectors)
    np.testing.assert_array_equal(batch, expected)
    np.testing.assert_array_equal(disagreement, expected_disagreement)
    np.testing.assert_array_equal(decoder.decode_batch(detectors), expected)
    for index, event in enumerate(detectors):
        np.testing.assert_array_equal(decoder.decode_one(event), expected[index])
        single, differs = decoder.prediction_and_disagreement(event)
        np.testing.assert_array_equal(single, expected[index])
        assert differs == disagreement[index]
    empty_batch, empty_mask = decoder.predictions_and_disagreement(detectors[:0])
    assert empty_batch.shape == (0, 1)
    assert empty_mask.shape == (0,)
    with pytest.raises(ValueError, match="matching the model"):
        decoder.decode_batch(detectors[:, :-1])


class _Recorder:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.uint8)
        self.batches = []

    def decode_batch(self, detectors):
        shot_ids = detectors[:, 0]
        self.batches.append(shot_ids.tolist())
        return self.predictions[shot_ids, None]


def test_batch_majority_omits_only_decided_shots():
    detectors = np.arange(4, dtype=np.uint8)[:, None]
    members = tuple(_Recorder(predictions) for predictions in (
        [0, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
    ))
    decoder = EnsembleDecoder(members, 1)
    np.testing.assert_array_equal(decoder.decode_batch(detectors)[:, 0], [0, 1, 1, 1])
    assert [member.batches for member in members] == [
        [[0, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 2, 3]], [[2, 3]], [[3]],
    ]
    prediction, mask = decoder.predictions_and_disagreement(detectors)
    np.testing.assert_array_equal(prediction[:, 0], [0, 1, 1, 1])
    np.testing.assert_array_equal(mask, [True, False, True, True])
    assert members[-1].batches[-1] == [1, 3]


@pytest.mark.parametrize("spec", [
    (),
    (EnsembleMember(), EnsembleMember("clock")),
    (EnsembleMember(), EnsembleMember(), EnsembleMember("clock")),
])
def test_invalid_voters_are_rejected(spec):
    with pytest.raises(ValueError):
        EnsembleSpec(spec)
