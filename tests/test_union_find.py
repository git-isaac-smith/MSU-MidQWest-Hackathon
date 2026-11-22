import numpy as np
import pytest
import stim

from bascd.union_find import compile_union_find


def test_union_find_peels_weighted_boundary_chain_with_packed_rows():
    pytest.importorskip("stabilizer_qec")
    model = stim.DetectorErrorModel("""
        error(0.01) D0 D8
        error(0.02) D8 L0
    """)
    decoder = compile_union_find(model)
    packed = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    expected = np.array([[0], [1], [1], [0]], dtype=np.uint8)
    np.testing.assert_array_equal(decoder.decode_batch(packed), expected)
    assert decoder.decode_batch(packed[:0]).shape == (0, 1)
