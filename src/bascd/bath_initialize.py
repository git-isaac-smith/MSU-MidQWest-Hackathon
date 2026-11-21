"""Fit a dual-rail bath offline and export a check schedule and channel priors."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bath import ERASURE_PENALTY, FIT_BOUNDS, BathModel, BathParameters, fit_bath
from .measure import environment
from .records import save_csv, save_json

REFERENCE = "https://zenodo.org/records/24326452"


def channel_record(channel):
    real = channel.frame_correction.real.tolist()
    imag = channel.frame_correction.imag.tolist()
    paulis = channel.conditional_paulis
    return {
        "time": channel.time,
        "erasure_probability": channel.erasure_probability,
        "near_field_probability": channel.near_field_probability,
        "far_field_loss_probability": channel.far_field_loss_probability,
        "worst_case_rail_return": channel.worst_case_rail_return,
        "conditional_paulis": paulis,
        "unflagged_pauli_probability": channel.unflagged_pauli_probability,
        "balanced_cost": channel.balanced_cost,
        "decoder_priors_given_pass": {"bit_flip": paulis["X"] + paulis["Y"],
                                      "phase_flip": paulis["Z"] + paulis["Y"]},
        "frame_correction": {"real": real, "imag": imag},
    }


def calibration_data(args):
    if args.traces is not None:
        with np.load(args.traces, allow_pickle=False) as record:
            return np.asarray(record["times"], dtype=float), np.asarray(record["traces"], dtype=float), None
    rng = np.random.default_rng(args.seed)
    # A separate synthetic bath is the source of the observations. Its parameters
    # are used only after the fit to check the prediction and selected frame.
    truth = BathModel(BathParameters(*rng.uniform(FIT_BOUNDS[:, 0], FIT_BOUNDS[:, 1])), args.attachment)
    times = np.linspace(0, 100, 321)
    return times, truth.calibration_traces(times), truth


def initialize(args):
    times, traces, truth = calibration_data(args)
    if (times.ndim != 1 or traces.shape != (len(times), 3, 3) or len(times) < 25 or
            not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0) or
            not np.isfinite(traces).all()):
        raise ValueError("calibration requires at least 25 increasing nonnegative times and a finite (time, 3, 3) trace")
    initial = None
    if args.initial is not None:
        initial = BathParameters(**json.loads(args.initial.read_text(encoding="utf-8-sig")))
    holdout = np.arange(len(times)) % 5 == 4
    fit = fit_bath(times[~holdout], traces[~holdout], attachment=args.attachment, starts=args.starts,
                   max_evaluations=args.max_evaluations, seed=args.seed, initial=initial)
    model = BathModel(fit.parameters, args.attachment)
    predicted = model.calibration_traces(times)
    clock, selected = model.initialize(args.clock, args.window_factor, args.points)
    check_end = args.clock * args.window_factor
    report = {
        "format_version": 1,
        "reference": REFERENCE,
        "environment": environment(),
        "model": "layered dual rail with vacuum loss and no thermal absorption",
        "attachment": args.attachment,
        "time_unit": "dimensionless source model time",
        "calibration_source": "exact synthetic traces" if truth else str(args.traces),
        "trace_sha256": hashlib.sha256(times.tobytes() + traces.tobytes()).hexdigest(),
        "calibration_span": [float(times[0]), float(times[-1])],
        "training_points": int((~holdout).sum()),
        "held_out_points": int(holdout.sum()),
        "holdout_method": "every fifth time sample of this record",
        "fit": asdict(fit),
        "held_out_trace_rmse": float(np.sqrt(np.mean((predicted[holdout] - traces[holdout])**2))),
        "fit_seed": args.seed,
        "fit_starts_budget": args.starts,
        "fit_max_evaluations_per_stage": args.max_evaluations,
        "initial_estimate": asdict(initial) if initial is not None else None,
        "check_window": [args.clock, check_end],
        "check_grid_points": args.points,
        "erasure_penalty": ERASURE_PENALTY,
        "extrapolates_beyond_calibration": bool(args.clock < times[0] or check_end > times[-1]),
        "clock": channel_record(clock),
        "selected": channel_record(selected),
    }
    if truth is not None:
        oracle_clock, oracle = truth.initialize(args.clock, args.window_factor, args.points)
        report["synthetic_check"] = {
            "clock_oracle_frame": channel_record(oracle_clock),
            "oracle_selected": channel_record(oracle),
            "selected_time_on_true_model_with_fitted_frame": channel_record(
                truth.channel(selected.time, frame_correction=selected.frame_correction)),
        }

    check_grid = np.linspace(args.clock, check_end, args.points)
    rows = []
    for t in check_grid:
        channel = model.channel(t)
        rows.append({"time": float(t), "rail_return": 1 - channel.erasure_probability,
                     "near_field_probability": channel.near_field_probability,
                     "far_field_loss_probability": channel.far_field_loss_probability,
                     "worst_case_rail_return": channel.worst_case_rail_return,
                     "balanced_cost": channel.balanced_cost})
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "bath-profile.json", report)
    save_csv(args.out / "check-window.csv", rows)
    np.savez_compressed(args.out / "calibration.npz", times=times, traces=traces)
    print(f"Check t={selected.time:.6g} from clock t={clock.time:.6g}; "
          f"held-out trace RMSE {report['held_out_trace_rmse']:.3g}")
    print(args.out / "bath-profile.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--traces", type=Path, help="NPZ calibration traces; omit for synthetic demo")
    parser.add_argument("--initial", type=Path, help="optional six-parameter estimate JSON")
    parser.add_argument("--attachment", choices=("one", "split"), default="split")
    parser.add_argument("--clock", type=float, default=4)
    parser.add_argument("--window-factor", type=float, default=2.5)
    parser.add_argument("--points", type=int, default=151)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    try:
        initialize(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
