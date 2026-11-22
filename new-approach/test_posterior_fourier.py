import numpy as np

from logical_reference import build_reference
from posterior_decoder import PosteriorDecoder
from stim_data import ROUNDS


def _walsh_transform(values):
    """In-place Walsh transform over the packed binary state index."""
    for stage in range(values.shape[1].bit_length() - 1):
        pairs = values.reshape(len(values), -1, 2, 1 << stage)
        left = pairs[:, :, 0].copy()
        right = pairs[:, :, 1]
        pairs[:, :, 0] = left + right
        right[:] = left - right


def _forward_probabilities(reference, observed, terminal, p_data, p_readout):
    """Compute both terminal probabilities by forward Walsh convolution."""
    check_count = np.count_nonzero(reference.check_mask)
    check_bits = 1 << np.arange(check_count)
    logical_bit = 1 << check_count
    states = np.arange(2 * logical_bit)
    popcount = np.asarray([int(state).bit_count() for state in states])
    sites = np.eye(reference.data_ids.size, dtype=bool).reshape(-1, *reference.data_ids.shape)
    site_shifts = reference.terminal_checks(sites) @ check_bits
    site_shifts += logical_bit * reference.observable_parity(sites)

    # Each odd overlap contributes the characteristic function of a single
    # Bernoulli X fault. The product gives the round transition spectrum.
    odd_overlap = popcount[states[:, None] & site_shifts] & 1
    data_rate = np.longdouble(p_data)
    spectrum = (1 - 2 * data_rate) ** odd_overlap.sum(axis=1)

    recorded = observed[:, :, reference.check_mask] @ check_bits
    syndrome = states & (logical_bit - 1)
    readout_rate = np.longdouble(p_readout)
    weights = np.zeros((len(observed), len(states)), dtype=np.longdouble)
    weights[:, 0] = 1
    for time in range(ROUNDS):
        _walsh_transform(weights)
        weights *= spectrum
        _walsh_transform(weights)
        flips = popcount[recorded[:, time, None] ^ syndrome]
        weights *= readout_rate**flips * (1 - readout_rate) ** (check_count - flips)
        # The inverse Walsh scale is common to all states and cancels here.
        weights /= weights.sum(axis=1, keepdims=True)

    final_syndrome = terminal @ check_bits
    final_states = final_syndrome[:, None] + np.array([0, logical_bit])
    final_mass = np.take_along_axis(weights, final_states, axis=1)
    return final_mass / final_mass.sum(axis=1, keepdims=True)


def test_distance_five_posterior_agrees_with_walsh_forward_recursion():
    p_data, p_readout = 0.008, 0.04
    reference = build_reference(5, p_data, p_readout)
    records = reference.circuit.compile_detector_sampler(seed=63199).sample(48)
    observed, terminal = reference.public_checks(records)

    # Include a fault-free record, a persistent single-site X that flips L0,
    # and a record with substantial weight in both logical classes.
    sites = np.eye(reference.data_ids.size, dtype=bool).reshape(-1, *reference.data_ids.shape)
    logical_site = np.flatnonzero(reference.observable_parity(sites))[0]
    extra_checks = np.zeros((3, ROUNDS, *reference.check_mask.shape), dtype=bool)
    extra_terminal = np.zeros((3, terminal.shape[1]), dtype=bool)
    extra_terminal[1] = reference.terminal_checks(sites[[logical_site]])[0]
    extra_checks[:, :, reference.check_mask] = extra_terminal[:, None, :]
    for time, row, column in ((1, 4, 0), (2, 1, 3), (2, 2, 2), (3, 2, 4), (3, 3, 1), (3, 4, 2)):
        extra_checks[2, time:, row, column] = True
    extra_terminal[2] = extra_checks[2, -1, reference.check_mask]
    observed = np.concatenate((extra_checks, observed))
    terminal = np.concatenate((extra_terminal, terminal))

    expected = _forward_probabilities(reference, observed, terminal, p_data, p_readout)
    p_one = PosteriorDecoder(5, p_data, p_readout).predict_proba(observed, terminal)
    np.testing.assert_allclose(np.column_stack((1 - p_one, p_one)), expected, atol=2e-10, rtol=2e-9)
    assert 0 < expected[0, 1] < 0.01
    assert 0 < expected[1, 0] < 0.01
    assert 0.4 < expected[2, 1] < 0.6
