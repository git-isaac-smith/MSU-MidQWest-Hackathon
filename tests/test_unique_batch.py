import numpy as np
import pytest

from bascd.unique_batch import UniqueBatchDecoder


def _prediction(detectors):
    parity = detectors.sum(axis=1, dtype=np.int64) % 2
    count_parity = np.count_nonzero(detectors, axis=1) % 2
    return np.column_stack((parity, count_parity)).astype(np.uint8)


class _Recorder:
    def __init__(self):
        self.batches = []
        self.singles = []

    def decode_batch(self, detectors):
        self.batches.append(detectors)
        return _prediction(detectors)

    def decode_one(self, detectors):
        self.singles.append(detectors)
        return _prediction(detectors[None, :])[0]


def test_deduplication_restores_row_order_across_chunks():
    types = np.zeros((4, 18), dtype=bool)
    types[[1, 3], 0] = True
    types[[2, 3], 16] = True
    row_order = [2, 0, 2, 1, 0, 3, 1, 3, 2, 3, 0, 0]
    detectors = types[row_order, ::2]
    detectors.flags.writeable = False
    recorder = _Recorder()
    decoder = UniqueBatchDecoder(recorder, 9, chunk_size=5, probe_shots=5)

    np.testing.assert_array_equal(decoder.decode_batch(detectors), _prediction(detectors))
    assert [len(batch) for batch in recorder.batches] == [3, 3, 1]
    for batch, indices in zip(recorder.batches, ([2, 0, 1], [3, 1, 2], [0]), strict=True):
        np.testing.assert_array_equal(batch, types[indices, ::2])
    np.testing.assert_array_equal(decoder.decode_batch(detectors), _prediction(detectors))
    assert [len(batch) for batch in recorder.batches] == [3, 3, 1, 3, 3, 1]


def test_high_entropy_probe_bypasses_chunking_and_gathering():
    detectors = np.arange(8, dtype=np.uint8)[:, None]
    detectors[-2:] = detectors[0]
    recorder = _Recorder()
    decoder = UniqueBatchDecoder(recorder, 1, chunk_size=2, probe_shots=4)
    np.testing.assert_array_equal(decoder.decode_batch(detectors), _prediction(detectors))
    assert len(recorder.batches) == 1
    assert recorder.batches[0] is detectors


def test_integer_detector_rows_are_compared_without_coercion():
    detectors = np.array([[2, 1], [1, 1], [2, 1], [2, 2]], dtype=np.uint8)
    recorder = _Recorder()
    decoder = UniqueBatchDecoder(recorder, 2)
    np.testing.assert_array_equal(decoder.decode_batch(detectors), _prediction(detectors))
    np.testing.assert_array_equal(recorder.batches[0], detectors[[0, 1, 3]])


def test_empty_inputs_and_single_shot_delegation():
    recorder = _Recorder()
    decoder = UniqueBatchDecoder(recorder, 0)
    detectors = np.zeros((4, 0), dtype=bool)
    np.testing.assert_array_equal(decoder.decode_batch(detectors), _prediction(detectors))
    assert recorder.batches[0].shape == (1, 0)
    assert decoder.decode_batch(detectors[:0]).shape == (0, 2)
    single = detectors[0]
    np.testing.assert_array_equal(decoder.decode_one(single), [0, 0])
    assert recorder.singles[0] is single
    with pytest.raises(ValueError, match="matching the model"):
        decoder.decode_batch(np.zeros((3, 1), dtype=bool))
    with pytest.raises(ValueError, match="matching the model"):
        decoder.decode_one(detectors)
