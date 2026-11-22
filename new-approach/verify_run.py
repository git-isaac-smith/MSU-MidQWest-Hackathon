"""Replay the labeled Stim test split, reload the CNNs, and recount saved errors."""

import argparse
import json
from pathlib import Path

from main import predict_sites
import numpy as np
from stim_data import sample_shots


def verify(run_dir):
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = report["configuration"]
    replay = sample_shots(config["distance"], config["test_shots"], config["p_data"],
                          config["p_readout"], report["seeds"]["test"])
    with np.load(run_dir / "held-out.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["observed"], replay.observed)
        np.testing.assert_array_equal(archive["targets"], replay.targets)
        np.testing.assert_array_equal(archive["clean_for_diagnostics"], replay.clean)
        for name, row in zip(("raw", "fft"), report["test"], strict=True):
            scores, selected, ids = predict_sites(run_dir, replay.observed, method=name)
            np.testing.assert_array_equal(ids, archive["data_ids"])
            # Float32 BLAS implementations can round the same convolution differently.
            np.testing.assert_allclose(scores, archive[f"{name}_scores"], rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(selected, archive[f"{name}_scores"] >= report["thresholds"][name])
            # Index 2 * truth + prediction gives TN, FP, FN, TP in that order.
            indices = 2 * replay.targets.astype(np.int8) + selected
            counts = np.bincount(indices.ravel(), minlength=4)
            np.testing.assert_array_equal(counts, [row[k] for k in ("tn", "fp", "fn", "tp")])
            np.testing.assert_allclose((counts[1] + counts[2]) / indices.size, row["site_error_rate"])
            print(f"{name} verified from {indices.size:,} site decisions, with {counts[1]} false calls and {counts[2]} misses.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    verify(parser.parse_args().run_dir)
