"""plotting module: training-curve parsing/plotting and the eval eps-vs-p plot."""
import json

from qecdec.plotting import plot_decode_time, plot_eps_vs_p, plot_training, read_log
from qecdec.utils import shot_and_round_rates, eps_round, wilson_interval


def _write_log(path, runs):
    with open(path, "w") as f:
        for run in runs:
            for step in run:
                f.write(json.dumps({"step": step, "loss": 0.5 / step, "train_ler": 0.1, "samples_per_s": 700.0,
                                    "lr": 1e-3}) + "\n")
                f.write(json.dumps({"step": step, "val": {"3": 0.05 / step, "5": 0.08 / step,
                                                          "mean": 0.065 / step}}) + "\n")


def test_read_log_keeps_last_run(tmp_path):
    _write_log(tmp_path / "log.jsonl", [[100, 200, 300], [100, 200]])
    train, val = read_log(str(tmp_path / "log.jsonl"))
    assert [r["step"] for r in train] == [100, 200]
    assert [r["step"] for r in val] == [100, 200] and val[0]["3"] > 0


def test_plot_training_writes_png(tmp_path):
    _write_log(tmp_path / "log.jsonl", [[100, 200, 300]])
    out = plot_training(str(tmp_path))
    assert out is not None and (tmp_path / "training_curves.png").stat().st_size > 1000


def test_plot_training_empty_log(tmp_path):
    (tmp_path / "log.jsonl").write_text("")
    assert plot_training(str(tmp_path)) is None


def test_plot_decode_time(tmp_path):
    recs = [{"d": d, "p": p, "decoder": n, "us_per_shot": us * d}
            for d in (3, 5) for p in (0.001, 0.003) for n, us in (("MWPM", 20.0), ("Hybrid", 300.0))]
    out = tmp_path / "t.png"
    plot_decode_time(str(out), ["MWPM", "Hybrid"], recs, [3, 5], 1024)
    assert out.stat().st_size > 1000


def test_plot_eps_vs_p_with_zero_error_point(tmp_path):
    records = []
    for name, k in (("MWPM", 40), ("Hybrid", 0)):
        for p in (0.001, 0.003):
            n, d = 1000, 3
            P, P_err, eps, eps_err = shot_and_round_rates(k, n, d)
            lo, hi = wilson_interval(k, n)
            records.append({"d": d, "p": p, "decoder": name, "errors": k, "eps": eps,
                            "eps_lo": eps_round(lo, d), "eps_hi": eps_round(hi, d), "outside_train_d": False})
    out = tmp_path / "e.png"
    plot_eps_vs_p(str(out), ["MWPM", "Hybrid"], records, [3], 1000)
    assert out.stat().st_size > 1000
