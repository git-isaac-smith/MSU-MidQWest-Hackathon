"""Compare the Stim tutorial's ordinary PyMatching decoder against the trained FFT CNN."""

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bascd.stats import paired_ler_difference, wilson95
from frequency import Passband, features
from localizer import Localizer
from logical_reference import build_reference
from matching_reference import MatchingReference
from reference_figure import plot_percent_comparison
from scoring import paired_site_difference


TUTORIAL = "https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb"
BATCH = 2048


def percentage_row(metric, method, decisions, truth):
    if decisions.shape != truth.shape or not truth.size:
        raise ValueError("Predictions and truth must cover the same nonempty array")
    errors = int(np.count_nonzero(decisions != truth))
    return dict(metric=metric, method=method, errors=errors, total=truth.size,
                error_percent=100 * errors / truth.size)


def compare(run_dir, out, logical_shots, seed):
    if logical_shots < 1:
        raise ValueError("Logical comparison needs at least one shot")
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = report["configuration"]
    physical = (config["distance"], config["p_data"], config["p_readout"])
    ref = build_reference(*physical)
    checkpoint = run_dir / "fft-cnn.npz"
    cnn = Localizer.load(checkpoint)
    band = Passband(**report["passband"])
    cutoff = report["thresholds"]["fft"]

    def fft_sites(observed):
        return cnn.predict(features(observed, ref.check_mask, band)) >= cutoff

    out.mkdir(parents=True, exist_ok=False)
    with np.load(run_dir / "held-out.npz", allow_pickle=False) as archive:
        observed = archive["observed"]
        np.testing.assert_array_equal(archive["data_ids"], ref.data_ids)
        site_matching = MatchingReference(*physical).predict(observed)
        site_fft = fft_sites(observed)
        np.testing.assert_array_equal(site_fft, archive["fft_scores"] >= cutoff)
        site_truth = archive["targets"]
    site_rows = [
        percentage_row("physical_site", "Site-tracking MWPM", site_matching, site_truth),
        percentage_row("physical_site", "FFT + CNN", site_fft, site_truth),
    ]
    site_delta = paired_site_difference(site_matching, site_fft, site_truth)

    # Both methods receive the same public detector rows, including terminal check parities.
    # Stim returns the withheld logical observable in a separate array.
    sampler = ref.circuit.compile_detector_sampler(seed=seed)
    logical_matching = np.empty(logical_shots, dtype=bool)
    logical_fft = np.empty_like(logical_matching)
    fft_five_rounds_only = np.empty_like(logical_matching)
    logical_truth = np.empty_like(logical_matching)
    packed = np.empty((logical_shots, (ref.circuit.num_detectors + 7) // 8), dtype=np.uint8)
    for start in range(0, logical_shots, BATCH):
        stop = min(start + BATCH, logical_shots)
        batch = slice(start, stop)
        detectors, actual = sampler.sample(stop - start, separate_observables=True)
        check_history, terminal = ref.public_checks(detectors)
        logical_matching[batch] = ref.mwpm_parity(detectors)
        site_calls = fft_sites(check_history)
        fft_five_rounds_only[batch] = ref.observable_parity(site_calls)
        correction = ref.complete_fft(site_calls, terminal)
        logical_fft[batch] = ref.observable_parity(correction)
        logical_truth[batch] = actual[:, 0]
        packed[batch] = np.packbits(detectors, axis=1, bitorder="little")

    logical_rows = [
        percentage_row("logical_shot", "PyMatching (MWPM)", logical_matching, logical_truth),
        percentage_row("logical_shot", "FFT + CNN + final matching", logical_fft, logical_truth),
    ]
    matching_error = logical_matching != logical_truth
    fft_error = logical_fft != logical_truth
    logical_delta = paired_ler_difference(
        [logical_shots], [int(np.sum(matching_error & ~fft_error))], [int(np.sum(fft_error & ~matching_error))],
    ) if logical_shots > 1 else None
    five_rounds_only = percentage_row("logical_shot_without_terminal_input",
                                      "FFT + CNN alone", fft_five_rounds_only, logical_truth)
    rows = logical_rows + site_rows + [five_rounds_only]
    with (out / "error-percentages.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ref.circuit.to_file(out / "logical-benchmark.stim")
    np.savez_compressed(out / "paired-decisions.npz", logical_detector_bits_little=packed,
                        logical_truth=logical_truth, logical_mwpm=logical_matching, logical_fft=logical_fft,
                        logical_fft_five_rounds_only=fft_five_rounds_only,
                        site_truth=site_truth, site_mwpm=site_matching, site_fft=site_fft)
    metadata = {
        "baseline_source": TUTORIAL, "model_run": run_dir.name,
        "distance": config["distance"], "rounds": report["rounds"],
        "noise_model": report["noise_model"], "p_data": config["p_data"], "p_readout": config["p_readout"],
        "fft_threshold": cutoff, "fft_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "site_test_seed": report["seeds"]["test"], "logical_stim_seed": seed, "logical_sample_batch_size": BATCH,
        "logical_detectors": ref.circuit.num_detectors, "versions": {p: version(p) for p in ("stim", "pymatching", "numpy", "scipy")},
        "logical_protocol": "Fresh Stim logical memory shots. Matching decodes the detector error model. FFT predicts five-round physical-site calls and corrects their residual ideal final check syndrome with unit-weight spatial matching. Both see the same public checks.",
        "site_protocol": "Original labeled five-round test shots. Both methods see only the five noisy check grids. MWPM tracks one minimum-weight fault history back to data sites. FFT uses its saved validation threshold.",
        "rows": rows, "logical_wilson95_percent": {
            row["method"]: [100 * v for v in wilson95(row["errors"], row["total"])] for row in logical_rows
        },
        "logical_paired_fft_minus_mwpm": logical_delta, "site_paired_fft_minus_mwpm": site_delta,
        "paired_difference_units": "Fractions. Multiply by 100 for percentage points.",
    }
    (out / "comparison.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_percent_comparison(out / "stim-vs-fft-error-percent.png", logical_rows, site_rows, five_rounds_only)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Saved five-round FFT training run")
    parser.add_argument("--out", type=Path, required=True, help="A new comparison directory")
    parser.add_argument("--logical-shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026112103)
    args = parser.parse_args()
    print(json.dumps(compare(args.run, args.out, args.logical_shots, args.seed), indent=2))
