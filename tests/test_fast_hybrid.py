import numpy as np
import pymatching
import pytest
import stim

from bascd.exact_small import compile_exact_small
from bascd.fast_hybrid import FastHybridDecoder, _PosteriorCache


def test_packed_exact_branch_preserves_joint_faults_and_padding():
    model = stim.DetectorErrorModel('error(0.2) D0 L0 ^ D1\nerror(0.1) D1\ndetector D2')
    reference = compile_exact_small(
        stim.DetectorErrorModel('error(0.2) D0 D1 L0\nerror(0.1) D1\ndetector D2'))
    decoder = FastHybridDecoder(model)
    records = np.arange(8, dtype=np.uint8)[:, None]
    unpacked = np.unpackbits(records, axis=1, count=3, bitorder='little').astype(bool)
    expected = reference.decode_batch(unpacked)
    np.testing.assert_array_equal(decoder.decode_batch(unpacked), expected)
    np.testing.assert_array_equal(decoder.decode_batch(records | 0xf8, bit_packed_shots=True,
                                                        bit_packed_predictions=True), expected)
    np.testing.assert_array_equal(decoder.decode_one(unpacked[3]), expected[3])


def test_sparse_cache_hits_have_exact_keys_and_misses_use_matching():
    model = stim.DetectorErrorModel('error(0.15) D0 L0\nerror(0.1) D1\nerror(0.1) D24')
    decoder = FastHybridDecoder(model, pair_fault_limit=2)
    records = np.zeros((6, 4), dtype=np.uint8)
    records[:, 0] = [0, 1, 2, 3, 3, 1]
    records[4:6, 3] = 1
    hits, values = decoder._cache.lookup(records)
    np.testing.assert_array_equal(hits, [0, 1, 2, 3])
    np.testing.assert_array_equal(values, [0, 1, 0, 1])
    baseline = pymatching.Matching.from_detector_error_model(model)
    expected = baseline.decode_batch(records, bit_packed_shots=True, bit_packed_predictions=True)
    padded = records | np.array([0, 0, 0, 128], dtype=np.uint8)
    np.testing.assert_array_equal(decoder.decode_batch(padded, bit_packed_shots=True), expected)
    assert decoder.decode_batch(records[:0], bit_packed_shots=True).shape == (0, 1)


def test_all_cache_inference_checks_full_keys_on_fingerprint_collision(monkeypatch):
    model = stim.DetectorErrorModel('error(0.15) D0 L0\nerror(0.1) D24')
    cache = _PosteriorCache(model, pair_fault_limit=2)
    records = np.array([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.uint8)
    from bascd import fast_hybrid
    fingerprint = fast_hybrid._fingerprint(records[:1])[0]
    monkeypatch.setattr(fast_hybrid, '_fingerprint',
                        lambda keys: np.full(len(keys), fingerprint, dtype=np.uint64))
    hits, values = cache.lookup(records)
    np.testing.assert_array_equal(hits, [0])
    np.testing.assert_array_equal(values, [1])


def test_detector_count_routes_only_hard_cache_misses():
    model = stim.DetectorErrorModel('error(0.15) D0 L0\nerror(0.1) D1\nerror(0.1) D24')
    decoder = FastHybridDecoder(model, pair_fault_limit=0, correlated_fallback=True,
                                correlated_min_detectors=3)

    class RecordingMatcher:
        def __init__(self, expected, **options):
            self.expected = expected
            self.options = options

        def decode_batch(self, shots, **options):
            np.testing.assert_array_equal(shots, self.expected)
            assert options == dict(bit_packed_shots=True, bit_packed_predictions=True, **self.options)
            return np.zeros((len(shots), 1), dtype=np.uint8)

    easy_miss = [1, 0, 0, 1]
    hard_miss = [3, 0, 0, 1]
    decoder._fallback = RecordingMatcher([easy_miss])
    decoder._correlated = RecordingMatcher([hard_miss], enable_correlations=True)
    records = np.array([hard_miss, [1, 0, 0, 0], easy_miss], dtype=np.uint8)
    expected = [[0], [1], [0]]
    np.testing.assert_array_equal(decoder.decode_batch(records, bit_packed_shots=True,
                                                        bit_packed_predictions=True), expected)


@pytest.mark.parametrize('records,packed', [(np.zeros((1, 2)), True),
                                             (np.array([[2, 0, 0]]), False),
                                             (np.zeros((1, 1), dtype=bool), True)])
def test_rejects_invalid_records(records, packed):
    decoder = FastHybridDecoder(stim.DetectorErrorModel('error(0.1) D0 L0\ndetector D2'))
    with pytest.raises(ValueError):
        decoder.decode_batch(records, bit_packed_shots=packed)
