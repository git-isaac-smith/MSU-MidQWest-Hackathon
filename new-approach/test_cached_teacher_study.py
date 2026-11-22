import numpy as np

from cached_teacher_study import CachedTeacher, PHYSICAL, PackedHistories
from logical_reference import build_reference


def test_key_preserves_exact_public_posterior_history_and_reversal():
    reference = build_reference(*PHYSICAL)
    det = reference.circuit.compile_detector_sampler(seed=56).sample(512)
    packed = np.packbits(det, axis=1, bitorder='little')
    history = PackedHistories(reference)
    direct, canonical = (history.keys(packed, reverse) for reverse in (False, True))
    obs, term = reference.public_checks(det)
    successor = obs.copy()
    successor[:, :4, reference.check_mask] = obs[:, 3::-1, reference.check_mask] ^ term[:, None, :]
    successor[:, -1] ^= reference.check_mask
    more = reference.detector_records(successor, term)
    more_packed = np.packbits(more, axis=1, bitorder='little')
    np.testing.assert_array_equal(history.keys(more_packed, reverse=True), canonical)
    assert np.any(history.keys(more_packed, reverse=False) != direct)
    # Changing the fifth noisy check alone preserves the full ordered key.
    successor = obs.copy()
    successor[:, -1] ^= reference.check_mask
    noise = reference.detector_records(successor, term)
    np.testing.assert_array_equal(history.keys(np.packbits(noise, axis=1, bitorder='little'), False), direct)


def test_nominal_fingerprint_collision_falls_back_to_mwpm():
    reference = build_reference(*PHYSICAL)
    zero = np.zeros((2, 9), dtype=np.uint8)
    one = np.zeros((1, 9), dtype=np.uint8)
    one[0, 0] = 1
    candidate = CachedTeacher(reference, (zero, np.ones(2, dtype=bool)), reverse=True, table_bits=0)
    records = np.concatenate((zero[:1], one))
    predictions, hits, routed = candidate.decode(records, diagnostics=True)
    np.testing.assert_array_equal(hits, [True, False])
    np.testing.assert_array_equal(routed, [False, False])
    np.testing.assert_array_equal(predictions, [1, candidate.mwpm(one)[0]])
