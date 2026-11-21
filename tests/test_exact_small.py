import itertools
import math

import numpy as np
import pytest
import stim

from bascd.exact_small import compile_exact_small


def exhaustive_joint(faults, detector_count):
    masses = [[[] for _ in range(2)] for _ in range(1 << detector_count)]
    for active in itertools.product((False, True), repeat=len(faults)):
        syndrome, logical = 0, 0
        weight = 1.0
        for present, (probability, mask, flip) in zip(active, faults):
            weight *= probability if present else 1 - probability
            if present:
                syndrome ^= mask
                logical ^= flip
        masses[syndrome][logical].append(weight)
    return np.array([[math.fsum(terms) for terms in row] for row in masses])


def all_syndromes(detector_count):
    indices = np.arange(1 << detector_count, dtype=np.uint32)
    return ((indices[:, None] >> np.arange(detector_count)) & 1).astype(bool)


@pytest.mark.parametrize("fair_probability", [0.13, 0.5])
def test_ml_matches_exhaustive_native_faults(fair_probability):
    # Two hyperedges share a syndrome with opposite logical effects. The
    # remaining faults cover repeated masks, deterministic and fair faults.
    model = stim.DetectorErrorModel(
        f"""
        error(0.10) D0 D1 D2 L0
        error(0.13) D0 D1 D2
        error(0.22) D0
        error({fair_probability}) D1 D2 L0
        error(0.92) D2 L0
        error(1) D1
        error(0.06) L0
        detector D3
        """
    )
    faults = [
        (0.10, 7, 1), (0.13, 7, 0), (0.22, 1, 0),
        (fair_probability, 6, 1), (0.92, 4, 1), (1, 2, 0), (0.06, 0, 1),
    ]
    exact = exhaustive_joint(faults, 4)
    syndromes = all_syndromes(4)
    decoder = compile_exact_small(model)
    predictions = decoder.decode_batch(syndromes)
    np.testing.assert_array_equal(predictions, [decoder.decode_one(s) for s in syndromes])
    robust = np.abs(exact[:, 0] - exact[:, 1]) > decoder.signed_mass_error_bound
    np.testing.assert_array_equal(predictions[robust, 0], np.argmax(exact[robust], axis=1))
    optimal_error = exact.min(axis=1).sum()
    actual_error = exact[np.arange(len(exact)), 1 - predictions[:, 0].astype(int)].sum()
    assert actual_error - optimal_error <= decoder.excess_ler_bound + 2e-16
    assert decoder.estimated_optimal_ler == pytest.approx(optimal_error, abs=2e-16)
    assert decoder.decode_batch(np.empty((0, 4), dtype=np.uint8)).shape == (0, 1)


def test_dem_repeat_shifts_detectors():
    model = stim.DetectorErrorModel(
        """
        repeat 2 {
            error(0.12) D0 L0
            shift_detectors 1
        }
        """
    )
    decoder = compile_exact_small(model)
    predictions = decoder.decode_batch(all_syndromes(2))
    np.testing.assert_array_equal(predictions[:, 0], [0, 1, 1, 0])
    with pytest.raises(ValueError, match="binary"):
        decoder.decode_batch(np.array([[0, 2]], dtype=np.uint8))


def test_lookup_uses_detector_bits_across_transform_chunks():
    model = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D17 L0")
    decoder = compile_exact_small(model)
    detectors = np.zeros((4, 18), dtype=bool)
    detectors[[1, 3], 0] = True
    detectors[[2, 3], 17] = True
    expected = [[0], [1], [1], [0]]
    np.testing.assert_array_equal(decoder.decode_batch(detectors), expected)
    np.testing.assert_array_equal([decoder.decode_one(shot) for shot in detectors], expected)


def test_logical_only_and_tie():
    model = stim.DetectorErrorModel("error(0.5) L0")
    decoder = compile_exact_small(model)
    assert decoder.estimated_optimal_ler == 0.5
    np.testing.assert_array_equal(decoder.decode_batch(np.empty((2, 0), dtype=bool)), [[0], [0]])


def test_rare_logical_mass_survives_cancellation():
    decoder = compile_exact_small(stim.DetectorErrorModel("error(1e-20) L0"))
    assert decoder.estimated_optimal_ler == pytest.approx(1e-20, abs=1e-34)
    assert decoder.excess_ler_bound == 0


def test_rejects_unbounded_or_mismatched_model():
    with pytest.raises(ValueError, match="24 detectors"):
        compile_exact_small(stim.DetectorErrorModel("error(0.1) D24 L0"))
    with pytest.raises(ValueError, match="one observable"):
        compile_exact_small(stim.DetectorErrorModel("error(0.1) D0 L1"))
    with pytest.raises(ValueError, match="undecomposed"):
        compile_exact_small(stim.DetectorErrorModel("error(0.1) D0 ^ D1 L0"))
