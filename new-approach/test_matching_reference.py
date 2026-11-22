import numpy as np
import pytest

from matching_reference import MatchingReference
from stim_data import ROUNDS, _layout, _simulate_batch, sample_shots


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_every_single_data_fault_has_a_one_site_syndrome_consistent_correction(distance):
    layout = _layout(distance)
    data_count = distance**2
    shots = ROUNDS * data_count
    data = np.eye(shots, dtype=bool).reshape(ROUNDS, data_count, shots)
    readout = np.zeros((ROUNDS, len(layout.ancilla_ids), shots), dtype=bool)
    batch = _simulate_batch(layout, data, readout)

    predicted = MatchingReference(distance, 0.01, 0).predict(batch.observed)
    assert predicted.dtype == bool
    np.testing.assert_array_equal(predicted.sum(axis=(1, 2)), 1)
    correction_faults = np.zeros_like(data)
    for time in range(ROUNDS):
        block = slice(time * data_count, (time + 1) * data_count)
        correction_faults[time, :, block] = predicted[block].reshape(data_count, data_count).T
    reproduced = _simulate_batch(layout, correction_faults, readout)
    np.testing.assert_array_equal(reproduced.observed, batch.observed)

    # Some boundary sites are genuinely indistinguishable even with ideal readout.
    same_syndrome = np.all(
        batch.observed[:data_count, None] == batch.observed[None, :data_count], axis=(2, 3, 4),
    )
    unique = np.flatnonzero(same_syndrome.sum(axis=1) == 1)
    assert unique.size
    np.testing.assert_array_equal(predicted[unique], batch.targets[unique])
    pairs = np.argwhere(np.triu(same_syndrome, k=1))
    assert pairs.size
    np.testing.assert_array_equal(predicted[pairs[:, 0]], predicted[pairs[:, 1]])


def test_open_final_boundary_weights_data_and_readout_faults_separately():
    layout = _layout(3)
    data = np.zeros((ROUNDS, 9, 2), dtype=bool)
    readout = np.zeros((ROUNDS, len(layout.ancilla_ids), 2), dtype=bool)
    # A final-round X at the top left has the same record as a readout error
    # on its single neighboring Z check. An earlier readout error has two events.
    data[-1, 0, 0] = True
    check_site = (1, 1)
    check_offset = np.ravel_multi_index(check_site, layout.geometry.check_mask.shape)
    active_offsets = np.flatnonzero(layout.geometry.check_mask)
    check_index = np.flatnonzero(active_offsets == check_offset).item()
    ancilla = layout.check_records[0, check_index]
    readout[-1, ancilla, 1] = True
    batch = _simulate_batch(layout, data, readout)
    np.testing.assert_array_equal(batch.observed[0], batch.observed[1])

    data_preferred = MatchingReference(3, 0.04, 0.01).predict(batch.observed)
    np.testing.assert_array_equal(data_preferred.sum(axis=(1, 2)), 1)
    readout_preferred = MatchingReference(3, 0.01, 0.04).predict(batch.observed)
    assert not readout_preferred.any()

    data[:] = False
    readout[:] = False
    readout[0, ancilla] = True
    early = _simulate_batch(layout, data, readout)
    # One readout event costs less than the two data events needed to mimic it.
    assert not MatchingReference(3, 0.04, 0.01).predict(early.observed).any()


@pytest.mark.parametrize(("p_data", "p_readout"), [(0, 0), (0, 0.04), (0, 1), (1, 0.04), (1, 1)])
def test_deterministic_data_predictions_at_allowed_probability_endpoints(p_data, p_readout):
    batch = sample_shots(3, 32, p_data, p_readout, 21)
    decoder = MatchingReference(3, p_data, p_readout)
    np.testing.assert_array_equal(decoder.predict(batch.observed), batch.targets)
    assert decoder.predict(batch.observed[:0]).shape == (0, 3, 3)


def test_more_likely_faulted_frame_is_corrected_by_matching_rare_deviations():
    batch = sample_shots(5, 128, 0.02, 0.06, 43)
    # Complementing independent data and readout fault draws adds the frame
    # observed when every opportunity faults and negates final data parity.
    frame = sample_shots(5, 1, 1, 1, 43)
    predicted = MatchingReference(5, 0.02, 0.06).predict(batch.observed)
    complemented = MatchingReference(5, 0.98, 0.94).predict(batch.observed ^ frame.observed)
    np.testing.assert_array_equal(complemented, predicted ^ frame.targets)
