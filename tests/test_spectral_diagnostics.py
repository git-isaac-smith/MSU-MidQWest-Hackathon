import csv

import numpy as np
import pytest
import stim

from bascd.spectral_diagnostics import summarize


def test_temporal_diagnostics_separate_ensemble_mean_and_shot_power(tmp_path):
    circuit = stim.Circuit("""
        DETECTOR(2, 2, 0)
        DETECTOR(2, 2, 1)
        DETECTOR(2, 2, 2)
        DETECTOR(2, 2, 3)
    """)
    records = np.array([[0b1101], [0b1010]], dtype=np.uint8)
    output = summarize(tmp_path, circuit, records)
    with output["detector_means"].open(newline="") as file:
        means = list(csv.DictReader(file))
    assert [float(row["event_probability"]) for row in means] == [0.5, 0.5, 0.5, 1.0]
    assert [row["data_readout_closure"] for row in means] == ["0", "0", "0", "1"]
    with output["temporal_power"].open(newline="") as file:
        dc, one_cycle = list(csv.DictReader(file))
    assert float(dc["mean_shot_power"]) == pytest.approx(0)
    assert float(one_cycle["frequency_cycles_per_round"]) == pytest.approx(1 / 3)
    assert float(one_cycle["mean_shot_power"]) == pytest.approx(2 / 9)
    assert float(one_cycle["mean_trace_power"]) == pytest.approx(0)
    assert float(one_cycle["shot_noise_in_mean_estimate"]) == pytest.approx(2 / 9)
