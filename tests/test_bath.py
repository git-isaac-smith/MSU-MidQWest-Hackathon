import numpy as np
import pytest

from bascd.bath import BathModel, BathParameters


@pytest.mark.parametrize("attachment", ["one", "split"])
def test_bath_starts_with_identity_logical_transfer(attachment):
    model = BathModel(BathParameters(0.7, 0.3, 0.4, 0.2, 0.08, 0.19), attachment=attachment)

    transfer = model.rail_transfer([0.0])
    assert transfer.shape == (1, 2, 2)
    np.testing.assert_allclose(transfer[0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        model.calibration_traces([0.0]),
        [[[1, 0, 0], [0, 0, 1], [0, 0, -1]]],
        atol=1e-12,
    )
    channel = model.channel(0.0)
    assert channel.erasure_probability == pytest.approx(0)
    assert channel.worst_case_rail_return == pytest.approx(1)
    assert channel.conditional_paulis == pytest.approx({"I": 1, "X": 0, "Y": 0, "Z": 0})


@pytest.mark.parametrize("attachment", ["one", "split"])
def test_bath_channel_pairs_erasure_with_corrected_logical_probabilities(attachment):
    model = BathModel(BathParameters(0.7, 0.3, 0.4, 0.2, 0.08, 0.19), attachment=attachment)
    time = 0.7
    transfer = model.rail_transfer([time])[0]
    singular_values = np.linalg.svd(transfer, compute_uv=False)
    average_return = np.square(singular_values).mean()
    assert 0 < average_return < 1

    channel = model.channel(time)
    assert channel.erasure_probability == pytest.approx(1 - average_return)
    assert channel.near_field_probability > 0
    assert channel.far_field_loss_probability > 0
    assert channel.erasure_probability == pytest.approx(
        channel.near_field_probability + channel.far_field_loss_probability
    )
    assert channel.worst_case_rail_return == pytest.approx(singular_values[-1] ** 2)
    assert set(channel.conditional_paulis) == {"I", "X", "Y", "Z"}
    assert all(0 <= value <= 1 for value in channel.conditional_paulis.values())
    assert sum(channel.conditional_paulis.values()) == pytest.approx(1)
    assert channel.unflagged_pauli_probability == pytest.approx(
        average_return * sum(channel.conditional_paulis[pauli] for pauli in "XYZ")
    )

    correction = channel.frame_correction
    np.testing.assert_allclose(correction.conj().T @ correction, np.eye(2), atol=1e-12)
    corrected = correction @ transfer
    corrected *= np.exp(-1j * np.angle(np.trace(corrected)))
    np.testing.assert_allclose(corrected, corrected.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(corrected), singular_values[::-1], atol=1e-12)

    uncorrected = model.channel(time, frame_correction=np.eye(2))
    assert uncorrected.erasure_probability == pytest.approx(channel.erasure_probability)
    assert uncorrected.unflagged_pauli_probability >= channel.unflagged_pauli_probability - 1e-12
    with pytest.raises(ValueError, match="unitary"):
        model.channel(time, frame_correction=np.zeros((2, 2)))


def test_bath_initialization_uses_a_nonzero_time_in_the_window():
    model = BathModel(BathParameters(0.7, 0.3, 0.4, 0.2, 0.08, 0.19))
    clock_time = 1.1
    window_factor = 2.5

    clock, selected = model.initialize(clock_time, window_factor=window_factor, points=201)

    assert clock.time == pytest.approx(clock_time)
    assert clock_time <= selected.time <= clock_time * window_factor
    expected = model.channel(selected.time)
    assert selected.erasure_probability == pytest.approx(expected.erasure_probability)
    assert selected.conditional_paulis == pytest.approx(expected.conditional_paulis)
    clock_cost = sum(clock.conditional_paulis[p] for p in "XYZ") + clock.erasure_probability / 8
    selected_cost = sum(selected.conditional_paulis[p] for p in "XYZ") + selected.erasure_probability / 8
    assert selected_cost <= clock_cost + 1e-12
