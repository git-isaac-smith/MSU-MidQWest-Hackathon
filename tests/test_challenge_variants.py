import numpy as np
import pytest

from bascd.challenge_variants import VARIANT_NAMES, generate_variant_circuit
from bascd.circuits import memory_circuit


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("basis", ["x", "z"])
def test_variants_keep_detector_measurement_layout(distance, basis):
    p = 0.005
    ordinary = memory_circuit(basis, distance, distance, p)
    for name in VARIANT_NAMES:
        variant = generate_variant_circuit(
            name, basis=f"memory_{basis}", distance=distance, rounds=distance, p=p,
        )
        assert variant.without_noise().flattened() == ordinary.without_noise().flattened()
        assert variant.get_detector_coordinates() == ordinary.get_detector_coordinates()
        assert variant.num_measurements == ordinary.num_measurements
        assert variant.num_observables == ordinary.num_observables
        assert variant.detector_error_model(decompose_errors=True).num_detectors == ordinary.num_detectors
        if name == "ordinary":
            assert variant == ordinary
        else:
            assert variant.flattened() != ordinary.flattened()


def test_final_round_measurement_changes_only_last_ancilla_readout():
    p = 0.005
    rounds = 5
    ordinary = memory_circuit("z", 3, rounds, p).flattened()
    variant = generate_variant_circuit(
        "final_round_measurement", distance=3, rounds=rounds, p=p,
    )
    variant_ops = list(variant)
    final_readout = max(i for i, op in enumerate(variant_ops) if op.name == "MR")
    added = variant_ops.pop(final_readout - 1)
    assert added.name == "X_ERROR"
    assert added.gate_args_copy() == [4 * p]
    assert added.targets_copy() == variant_ops[final_readout - 1].targets_copy()
    assert variant_ops == list(ordinary)


@pytest.mark.parametrize(
    ("name", "expected_rounds", "expected_pair_distance", "paulis"),
    [
        ("measurement_burst", {2}, 8, {(True, False)}),
        ("spatial_correlated", set(range(5)), 4, {(True, False), (False, True)}),
    ],
)
def test_joint_events_connect_neighbors(name, expected_rounds, expected_pair_distance, paulis):
    variant = generate_variant_circuit(name, distance=5, rounds=5, p=0.005)
    coords = variant.get_final_qubit_coordinates()
    round_count = 0
    seen_rounds = set()
    seen_paulis = set()
    for op in variant:
        if op.name == "MR":
            round_count += 1
        if op.name == "E":
            first, second = op.targets_copy()
            squared_distance = np.sum((np.array(coords[first.value]) - coords[second.value]) ** 2)
            assert squared_distance == expected_pair_distance
            assert (first.is_x_target, first.is_z_target) == (second.is_x_target, second.is_z_target)
            seen_rounds.add(round_count)
            seen_paulis.add((first.is_x_target, first.is_z_target))
    assert seen_rounds == expected_rounds
    assert seen_paulis == paulis
