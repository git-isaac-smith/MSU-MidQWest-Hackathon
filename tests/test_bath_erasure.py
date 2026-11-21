from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from bascd.bath_erasure import BathNoise, LocationDecoder, ZMemory, main, run_control


PROFILE = Path(__file__).resolve().parents[1] / "experiments/iteration-03/bath-calibration/bath-profile.json"


def test_rotated_memory_fault_map_has_independent_geometric_reference():
    code = ZMemory.rotated(3)
    assert code.sites == (1, 3, 5, 8, 10, 12, 15, 17, 19)
    supports = ({8, 15}, {1, 3, 8, 10}, {10, 12, 17, 19}, {5, 12})
    expected_checks = [[int(site in support) for site in code.sites] for support in supports]
    logical_top_row = [int(site in {1, 3, 5}) for site in code.sites]
    x_faults = np.eye(len(code.sites), dtype=np.uint8)
    for pauli in (1, 2):
        syndrome, truth = code.measure(pauli * x_faults)
        np.testing.assert_array_equal(syndrome, np.asarray(expected_checks).T)
        np.testing.assert_array_equal(truth[:, 0], logical_top_row)
    syndrome, truth = code.measure(3 * x_faults)
    assert not syndrome.any()
    assert not truth.any()


def test_clicks_sample_the_pass_channel_and_uniform_replacements():
    noise = BathNoise(1.0, 0.4, (0.6, 0.2, 0.15, 0.05))
    paulis, flags = noise.sample_paulis(30_000, 9, np.random.default_rng(17))
    assert flags.dtype == np.bool_
    assert flags.mean() == pytest.approx(0.4, abs=0.004)
    np.testing.assert_allclose(np.bincount(paulis[flags]) / flags.sum(), [0.25] * 4, atol=0.005)
    np.testing.assert_allclose(
        np.bincount(paulis[~flags]) / (~flags).sum(), noise.conditional_paulis, atol=0.005,
    )
    flips = (paulis == 1) | (paulis == 2)
    assert flips[~flags].mean() == pytest.approx(noise.passed_flip_probability, abs=0.005)
    assert flips.mean() == pytest.approx(noise.blind_flip_probability, abs=0.005)


def test_heralds_change_prediction_without_revealing_the_fault():
    code = ZMemory.rotated(3)
    decoder = LocationDecoder(code, BathNoise(1.0, 0.2, (0.97, 0.03, 0, 0)))
    site_indices = {site: index for index, site in enumerate(code.sites)}
    faults = np.zeros((2, len(code.sites)), dtype=np.uint8)
    # These private errors give the same check parities and opposite logical parities.
    faults[0, site_indices[8]] = 1
    faults[1, [site_indices[1], site_indices[15]]] = 1
    syndrome, truth = code.measure(faults)
    np.testing.assert_array_equal(syndrome[0], syndrome[1])
    np.testing.assert_array_equal(truth[:, 0], [0, 1])
    flags = np.broadcast_to(faults[1].astype(bool), faults.shape)
    np.testing.assert_array_equal(decoder.decode_blind(syndrome), [[0], [0]])
    np.testing.assert_array_equal(decoder.decode_heralded(syndrome, flags), [[1], [1]])
    np.testing.assert_array_equal(decoder.decode_heralded(syndrome, np.zeros_like(flags)), [[0], [0]])


@pytest.mark.parametrize("passed_probability", (0, 0.04, 0.5, 0.8, 1))
def test_matching_logical_prediction_can_come_from_an_exhaustive_minimum_cost_fault(passed_probability):
    code = ZMemory.rotated(3)
    noise = BathNoise(1.0, 0.2, (1 - passed_probability, passed_probability, 0, 0))
    decoder = LocationDecoder(code, noise)
    # Enumerate every X fault configuration and each of the 16 possible syndromes.
    site_bits = np.arange(len(code.sites))
    patterns = ((np.arange(2 ** len(site_bits))[:, None] >> site_bits) & 1).astype(np.uint8)
    syndrome, logical = code.measure(patterns)
    syndromes = np.unique(syndrome, axis=0)

    def check_minimum_fault(predictions, cost):
        for measured, predicted in zip(syndromes, predictions[:, 0], strict=True):
            eligible = np.all(syndrome == measured, axis=1)
            costs = cost[eligible]
            assert predicted in logical[eligible, 0][costs == costs.min()]

    blind_direction = np.sign(0.5 - noise.blind_flip_probability)
    check_minimum_fault(decoder.decode_blind(syndromes), blind_direction * patterns.sum(axis=1, dtype=int))
    for flagged in ((), (0, 6), (1, 3, 4, 8), tuple(site_bits)):
        flags = np.isin(site_bits, flagged)
        batch_flags = np.broadcast_to(flags, (len(syndromes), len(site_bits)))
        pass_direction = np.sign(0.5 - passed_probability)
        cost = pass_direction * patterns[:, ~flags].sum(axis=1, dtype=int)
        check_minimum_fault(decoder.decode_heralded(syndromes, batch_flags), cost)


def test_result_pairs_decoders_within_each_archived_clock():
    result = run_control(PROFILE, shots=2000, seed=31)
    assert result["distance"] == 3
    assert result["clocks"]["fixed"]["time"] == 4.0
    assert result["clocks"]["selected"]["time"] == 8.08
    fixed, selected = result["clocks"].values()
    assert fixed["seed"] != selected["seed"]
    for clock in (fixed, selected):
        paired = clock["paired"]
        blind_only = paired["blind_only_failed"]
        heralded_only = paired["heralded_only_failed"]
        assert paired["both_correct"] + blind_only + heralded_only + paired["both_failed"] == clock["shots"]
        assert paired["both_failed"] + blind_only == clock["blind"]["failures"]
        assert paired["both_failed"] + heralded_only == clock["heralded"]["failures"]
        assert paired["blind_minus_heralded_failure_fraction"] == (blind_only - heralded_only) / clock["shots"]
        p_value = binomtest(blind_only, blind_only + heralded_only).pvalue
        assert paired["exact_two_sided_log10_p_value"] == pytest.approx(np.log10(p_value))
    assert selected["heralded"]["failures"] < selected["blind"]["failures"]


def test_cli_refuses_to_overwrite_an_output(tmp_path, capsys):
    output = tmp_path / "result.json"
    main(["--profile", str(PROFILE), "--output", str(output), "--shots", "10"])
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main(["--profile", str(PROFILE), "--output", str(output), "--shots", "20"])
    assert output.read_bytes() == original
    assert "output already exists" in capsys.readouterr().err
