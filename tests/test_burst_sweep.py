import csv

import numpy as np

from bascd.burst_samples import BurstSampler
from bascd.burst_sweep import apply_events, event_draws, run
from bascd.circuits import memory_circuit


def test_two_hits_in_one_round_cancel_only_where_they_overlap():
    kernel = np.array([[.5, .1], [.1, .5]])
    draws = (np.array([0, 0]), np.array([0, 0]), np.array([0, 1]),
             np.array([[.2, .4], [.05, .25]]), np.array([2, 0]))
    actual = apply_events(draws, kernel, rounds=1)
    np.testing.assert_array_equal(actual[:, 0, :], [[False, True], [False, False]])


def test_one_uniform_core_and_common_random_lengths():
    geometry = BurstSampler(memory_circuit("z", 3, 3, .005), 3)
    draws = event_draws(np.random.default_rng(7), 600, geometry, .001, "one")
    np.testing.assert_array_equal(draws[0], np.arange(600))
    np.testing.assert_array_equal(draws[-1], np.ones(600))
    short = apply_events(draws, geometry.kernel(1.), 3)
    long = apply_events(draws, geometry.kernel(4.), 3)
    assert np.all(~short | long)
    assert np.all(long.any(axis=2).sum(axis=1) <= 1)


def test_zero_density_blind_predictions_match_background(tmp_path):
    folder = tmp_path / "sweep"
    run(folder, distances=(3,), lengths=(1., 4.), density=0., seeds=1, shots=175, batch=64)
    with np.load(folder / "seeds-d3-0.npz", allow_pickle=False) as z:
        assert not z["poisson_counts"].any()
        for mode in ("poisson", "independent"):
            for xi in (1, 4):
                np.testing.assert_array_equal(z[f"{mode}_{xi}"], z["none_0"])
        with (folder / "results.csv").open() as stream:
            for row in csv.DictReader(stream):
                key = "{}_{:g}".format(row["exposure"], float(row["xi"]))
                bits = np.unpackbits(z[key], bitorder="little", count=int(row["shots"]))
                assert int(row["errors"]) == bits.sum()
