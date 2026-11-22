"""Run a paired comparison of raw and Fourier-filtered five-round check histories."""

import argparse
import csv
import json
import os
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from figures import plot_errors, plot_example, plot_joint_spectrum
from frequency import WEIGHTS, Passband, example_spectra, features, retained_energy
from localizer import Localizer
from scoring import paired_site_difference, select_threshold, site_metrics
from stim_data import ROUNDS, geometry, sample_shots
from compare_stim_fft import compare


METHODS = {"raw": "Raw checks + CNN", "fft": "Filtered FFT + CNN"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--p-data", type=float, default=0.008)
    parser.add_argument("--p-readout", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--pretrain-shots", type=int, default=8000)
    parser.add_argument("--train-shots", type=int, default=16000)
    parser.add_argument("--validation-shots", type=int, default=4000)
    parser.add_argument("--test-shots", type=int, default=10000)
    parser.add_argument("--logical-shots", type=int, help="Fresh paired Stim logical shots (defaults to --test-shots)")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temporal-cutoff", type=float, default=0.20)
    parser.add_argument("--spatial-cutoff", type=float, default=0.48)
    args = parser.parse_args()
    for name in ("pretrain_shots", "train_shots", "validation_shots", "test_shots",
                 "pretrain_epochs", "epochs", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not 0 <= args.p_data <= 1 or not 0 <= args.p_readout <= 1 or args.seed < 0:
        parser.error("Fault probabilities must be in [0, 1] and the seed must be nonnegative")
    if args.logical_shots is not None and args.logical_shots < 1:
        parser.error("logical-shots must be positive")
    return args


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def predict_sites(run_dir, observed, method="fft"):
    """Infer from five public check grids per shot, yielding scores, calls and Stim ID grid."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    mode = {"raw": None, "fft": Passband(**report["passband"])}[method]
    layout = geometry(report["configuration"]["distance"])
    cnn = Localizer.load(run_dir / f"{method}-cnn.npz")
    scores = cnn.predict(features(observed, layout.check_mask, mode))
    return scores, scores >= report["thresholds"][method], layout.data_ids


def qubit_records(mask, layout, probabilities):
    return [
        {"stim_id": int(layout.data_ids[row, col]), "row": int(row), "column": int(col),
         "stim_xy": [2 * int(col) + 1, 2 * int(row) + 1],
         "cnn_score": float(probabilities[row, col])}
        for row, col in zip(*np.nonzero(mask))
    ]


def save_example(out, observed, targets, scores, threshold, layout, band):
    # Show a shot with a decoder decision. Ground truth is consulted after selecting it.
    calls = scores >= threshold
    flagged = np.flatnonzero(calls.any(axis=(1, 2)))
    idx = int(flagged[0]) if len(flagged) else 0
    frames, stack, filtered_stack, joint, mask = example_spectra(observed[idx], band)
    plot_example(out, observed[idx], frames, stack, filtered_stack, WEIGHTS, layout.data_ids,
                 targets[idx], scores[idx], threshold, idx, check_mask=layout.check_mask)
    plot_joint_spectrum(out, joint, mask)
    example = {
        "test_shot_zero_based": idx,
        "selection_rule": "First test shot with any FFT CNN selection, else shot zero. Independent of truth.",
        "injected_final_odd_parity": qubit_records(targets[idx], layout, scores[idx]),
        "fft_cnn_selected": qubit_records(calls[idx], layout, scores[idx]),
    }
    write_json(out / "example-qubits.json", example)
    return example


def run(args):
    layout = geometry(args.distance)
    band = Passband(args.temporal_cutoff, args.spatial_cutoff)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    names = ("pretrain", "train", "validation", "test", "initialization", "pretrain_order", "train_order")
    children = np.random.SeedSequence(args.seed).spawn(len(names))
    seeds = {name: int(child.generate_state(1, dtype=np.uint64)[0]) for name, child in zip(names, children)}

    def draw(split, readout):
        return sample_shots(args.distance, getattr(args, f"{split}_shots"), args.p_data, readout, seeds[split])

    pretrain = draw("pretrain", 0.0)
    base = Localizer(channels=7, seed=seeds["initialization"])
    pretrain_lr, finetune_lr = 0.003, 0.002
    histories = {"pretrain": base.fit(
        features(pretrain.observed, layout.check_mask), pretrain.targets,
        epochs=args.pretrain_epochs, batch_size=args.batch_size, lr=pretrain_lr, seed=seeds["pretrain_order"],
    )}
    base.save(out / "pretrained.npz")
    del pretrain
    train = draw("train", args.p_readout)
    validation = draw("validation", args.p_readout)
    thresholds, validation_metrics = {}, {}
    modes = {"raw": None, "fft": band}
    for name, mode in modes.items():
        cnn = Localizer.load(out / "pretrained.npz")
        histories[name] = cnn.fit(
            features(train.observed, layout.check_mask, mode), train.targets,
            epochs=args.epochs, batch_size=args.batch_size, lr=finetune_lr, seed=seeds["train_order"],
        )
        scores = cnn.predict(features(validation.observed, layout.check_mask, mode))
        thresholds[name] = select_threshold(scores, validation.targets)
        validation_metrics[name] = site_metrics(scores >= thresholds[name], validation.targets)
        cnn.save(out / f"{name}-cnn.npz")
    del train, validation

    # The final split is sampled only after both checkpoints and decision rules are fixed.
    test = draw("test", args.p_readout)
    scores, calls, rows = {}, {}, []
    for name, mode in modes.items():
        cnn = Localizer.load(out / f"{name}-cnn.npz")
        scores[name] = cnn.predict(features(test.observed, layout.check_mask, mode))
        calls[name] = scores[name] >= thresholds[name]
        rows.append({"method": METHODS[name], **site_metrics(calls[name], test.targets)})
    difference = paired_site_difference(calls["raw"], calls["fft"], test.targets)
    energy = {
        "clean_data_checks": retained_energy(test.clean, band),
        "readout_residual": retained_energy(test.observed.astype(np.int8) - test.clean.astype(np.int8), band),
    }
    np.savez_compressed(
        out / "held-out.npz", observed=test.observed, clean_for_diagnostics=test.clean, targets=test.targets,
        raw_scores=scores["raw"], fft_scores=scores["fft"], data_ids=layout.data_ids, check_mask=layout.check_mask,
    )
    example = save_example(out, test.observed, test.targets, scores["fft"], thresholds["fft"], layout, band)
    plot_errors(out, rows, paired=difference)
    with (out / "site-errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "configuration": {k: v for k, v in vars(args).items() if k != "out"},
        "noise_model": "Independent data X before each round and independent ancilla readout flips just before MR. Other operations ideal.",
        "rounds": ROUNDS, "active_z_checks": int(layout.check_mask.sum()), "seeds": seeds,
        "target": "Final odd X-fault parity per physical data site",
        "passband": asdict(band), "round_weights": WEIGHTS.tolist(),
        "transform": "Zero-pad spatial axes to twice their size. Even-reflect five rounds to ten. Retain DC and joint passband.",
        "cnn_channels": "Five raw or inverse-FFT check fields, their time-weighted sum and the fixed check mask",
        "pretrain_lr": pretrain_lr, "finetune_lr": finetune_lr, "epoch_losses": histories,
        "threshold_objective": "Maximum pooled positive-site F1 on validation",
        "thresholds": thresholds, "validation": validation_metrics, "test": rows,
        "paired_site_error_fft_minus_raw": difference, "fraction_fourier_energy_retained": energy,
        "versions": {name: version(name) for name in ("stim", "numpy", "scipy", "matplotlib")},
        "python": platform.python_version(),
    }
    write_json(out / "metrics.json", report)
    logical_seed = int(np.random.SeedSequence([args.seed, 72]).generate_state(1, dtype=np.uint64)[0])
    percentages = compare(out, out / "stim-comparison", args.logical_shots or args.test_shots, logical_seed)
    print(json.dumps({"stim_vs_fft": percentages, "cnn_control": rows, "paired_difference": difference,
                      "energy_retained": energy, "example": example}, indent=2))


if __name__ == "__main__":
    run(parse_args())
