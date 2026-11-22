import numpy as np
import pymatching
import pytest
import stim

from bascd.circuits import memory_circuit
from bascd.temporal_filter import TemporalFilterDecoder


@pytest.mark.parametrize("distance,basis", [(3, "x"), (5, "z"), (7, "x")])
def test_full_band_matches_raw_mwpm(distance, basis):
    circuit = memory_circuit(basis, distance, distance, 0.005)
    graph = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    shots = circuit.compile_detector_sampler(seed=19).sample(37, bit_packed=True)
    decoder = TemporalFilterDecoder(circuit, graph, cutoff=0.5)
    np.testing.assert_array_equal(
        decoder.decode_batch(shots),
        graph.decode_batch(shots, bit_packed_shots=True, bit_packed_predictions=True),
    )
    assert decoder.decode_batch(shots[:0]).shape == (0, 1)


def test_lowpass_respects_open_ends_and_missing_detector_slots():
    circuit = stim.Circuit("""
        DETECTOR(0, 0, 0)
        DETECTOR(0, 0, 1)
        DETECTOR(0, 0, 2)
        DETECTOR(0, 0, 4)
        DETECTOR(0, 0, 5)
        DETECTOR(0, 0, 6)
        DETECTOR(1, 0, 1)
        DETECTOR(1, 0, 2)
        OBSERVABLE_INCLUDE(0)
    """)
    graph = pymatching.Matching()
    for detector in range(circuit.num_detectors):
        graph.add_boundary_edge(detector, fault_ids={0}, weight=4)
    decoder = TemporalFilterDecoder(circuit, graph, cutoff=0, threshold=0.5)
    shots = np.array([[0b01100011], [0b01000100]], dtype=np.uint8)
    residual, correction = decoder.preprocess(shots)
    np.testing.assert_array_equal(residual[:, 0], [0b11100111, 0b11000110])
    np.testing.assert_array_equal(correction, np.zeros((2, 1), dtype=np.uint8))


def test_guarded_pairs_apply_observable_parity_and_leave_competing_events():
    circuit = stim.Circuit("""
        DETECTOR(0, 0, 0)
        DETECTOR(0, 0, 1)
        DETECTOR(0, 0, 2)
        DETECTOR(0, 0, 3)
        DETECTOR(1, 0, 1)
        DETECTOR(2, 0, 1)
        DETECTOR(2, 0, 2)
        OBSERVABLE_INCLUDE(0)
    """)
    graph = pymatching.Matching()
    for detector in range(circuit.num_detectors):
        graph.add_boundary_edge(detector, weight=4)
    graph.add_edge(1, 2, fault_ids={0}, weight=1)
    graph.add_edge(1, 4, weight=1)
    graph.add_edge(0, 1, fault_ids={0}, weight=1)
    graph.add_edge(5, 6, fault_ids={0}, weight=1)
    shots = np.array([[0b00000110], [0b00010110], [0b00000011], [0b01100110]], dtype=np.uint8)
    decoder = TemporalFilterDecoder(circuit, graph, mode="guarded")
    residual, correction = decoder.preprocess(shots)
    np.testing.assert_array_equal(residual[:, 0], [0, 0b00010110, 0b00000011, 0])
    np.testing.assert_array_equal(correction[:, 0], [1, 0, 0, 0])
    np.testing.assert_array_equal(
        decoder.decode_batch(shots),
        graph.decode_batch(residual, bit_packed_shots=True, bit_packed_predictions=True) ^ correction,
    )
