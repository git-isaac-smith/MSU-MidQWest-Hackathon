# Pipeline commands

All commands run from the project root. `python` is the interpreter that has the requirements installed.
Every script that writes results makes a new timestamped folder, so nothing is overwritten, except
`train.py`, which writes into the config's `out_dir` (see stage 3).

```
setup -> tests -> benchmark -> experiments -> select best -> train -> evaluate
  1        2         3 (opt)       5              6           7        8
                      train (default configs) = 4
```

## 1. Environment

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux / remote machine:

```
python3 -m venv ~/venvs/qec && source ~/venvs/qec/bin/activate
pip install -r requirements.txt
```

Check: `python -c "import torch, stim, pymatching, matplotlib; print(torch.__version__)"`

## 2. Tests

```
python -m pytest -q -m "not smoke"      # fast checks (geometry, data, models, plotting, experiments, selection)
python -m pytest -q -m smoke            # d=3 and tiny-sweep training checks, a few minutes on CPU
```

## 3. Speed benchmark (optional)

Time per training step and projected hours, per distance:

```
python benchmark.py --config configs/hybrid.yaml
python benchmark.py --config configs/cnn_baseline.yaml
python benchmark.py --config configs/cnn_baseline.yaml --set compile=true
```

## 4. Train with the default configs

```
python train.py --config configs/hybrid.yaml
python train.py --config configs/cnn_baseline.yaml
python train.py --config configs/smoke_d3.yaml                  # quick d=3 check
```

Overrides and options:

```
python train.py --config configs/hybrid.yaml --set steps=5000 p=0.004 d_set=[3,5] out_dir=runs/hybrid_short
python train.py --config configs/hybrid.yaml --set use_boundary_tokens=false out_dir=runs/hybrid_no_bt
python train.py --config configs/cnn_baseline.yaml --no-plot
```

- Output goes to `out_dir` from the config (`runs/hybrid`, `runs/cnn_baseline`): `config.json`, `log.jsonl`,
  `last.pt`, `best.pt` (lowest mean validation LER) and `training_curves.png`.
- Training again into the same `out_dir` overwrites the checkpoints and restarts from step 1: give each run its
  own `out_dir=`.
- There is no resume from a checkpoint. To pause, suspend the process (Windows: Resource Monitor, CPU tab,
  right-click `python.exe`, Suspend Process; Linux: `kill -STOP <pid>` and `kill -CONT <pid>`).
- Redraw the curves of an existing run: `python -m qecdec.plotting runs/hybrid`
- Long runs on a remote machine, surviving a closed SSH session:
  ```
  tmux new -s train        # then run the command, detach with Ctrl+b d, reattach with: tmux attach -t train
  nohup python train.py --config configs/cnn_baseline.yaml > cnn_train.log 2>&1 &
  ```

## 5. Hyperparameter experiments (short runs)

Plan and rough time estimate, no training:

```
python experiment.py --list
python experiment.py --list --model hybrid --groups lr ablation
```

Run (each run is one changed hyperparameter, 2000 steps by default; `--model` is `hybrid`, `cnn` or `both`):

```
python experiment.py                                                      # everything, both models
python experiment.py --model hybrid --groups baseline seed lr ablation
python experiment.py --model cnn --groups baseline seed lr width dilations --steps 3000
```

Groups: baseline, seed, distance, p, lr, shots, steps, weight_decay, grad_clip, width, boundary_clip,
ablation, attn_layers, heads, stem_layers, mlp_ratio, dilations.

Tune each distance on its own: one sweep per base distance, skipping the `distance` and `p` groups. `baseline` and
`seed` are needed as the reference and noise level for stage 6.

```
python experiment.py --d-set 3 --groups baseline seed lr weight_decay grad_clip width boundary_clip attn_layers heads stem_layers mlp_ratio dilations
python experiment.py --d-set 5 --groups baseline seed lr weight_decay grad_clip width boundary_clip attn_layers heads stem_layers mlp_ratio dilations
python experiment.py --d-set 7 --groups baseline seed lr width attn_layers dilations
```

Larger d is much slower (see `--list`). A base of `3 5` (the default) gives d=3 and d=5 results from one sweep.

Other options: `--val-shots N`, `--set key=value ...` (applies to every run), `--device cpu`, `--no-plot`,
`--out-dir DIR`.

Continue an interrupted sweep (finished runs are skipped, failed ones are retried):

```
python experiment.py --run-dir experiments/<timestamp> <same flags as before>
```

Output in `experiments/<timestamp>/`: one folder per run (`<model>__<group>-<label>/` with `config.json`,
`log.jsonl`, `best.pt`, `training_curves.png`) plus `results.csv` and `results.json`, rewritten after every run.

## 6. Pick the best hyperparameters per distance

```
python select_best.py                                        # newest sweep in experiments/
python select_best.py experiments/<timestamp>                # a specific sweep (repeat to analyse several)
python select_best.py experiments/<timestamp> --d 3 5 --sigma 1.5
python select_best.py experiments/<timestamp> --metric ler   # raw validation LER instead of LER / MWPM LER
python select_best.py experiments/<timestamp> --exclude-groups    # also allow ablation, steps, shots to be combined
```

Output in `best_hyperparameters/<timestamp>/`: `summary.txt`, `best_hyperparameters.json`, `per_group.csv` and
`configs/<model>_d<d>.yaml` (full-training config per model and distance, `d_set=[d]`, `out_dir=runs/best_<model>_d<d>`).
A value counts only if its gain over the baseline exceeds `--sigma` times the baseline/seed spread. The combined
config is untested as a whole, so confirm it in stage 7.

## 7. Train with the selected hyperparameters

```
python train.py --config best_hyperparameters/<timestamp>/configs/hybrid_d5.yaml
python train.py --config best_hyperparameters/<timestamp>/configs/cnn_d5.yaml
```

Quick confirmation of a combined config before the full run:

```
python train.py --config best_hyperparameters/<timestamp>/configs/hybrid_d5.yaml --set steps=3000 out_dir=runs/confirm_hybrid_d5
```

For a mixed-distance model, copy a config and set `d_set=[3,5,7]` (and a new `out_dir=`).

## 8. Evaluate and compare (MWPM vs CNN vs hybrid)

```
python eval.py                                                          # default checkpoints, MWPM + CNN + hybrid
python eval.py --hybrid runs/hybrid/best.pt --cnn runs/cnn_baseline/best.pt
python eval.py --hybrid runs/best_hybrid_d5/best.pt --cnn runs/best_cnn_d5/best.pt --d 5 --p 0.002 0.003 0.005
python eval.py --shots 50000 --d 3 5 7 9 --p 0.001 0.003 --batch-size 512
python eval.py --device cpu --no-plot
```

- Defaults: `--shots 20000`, `--p 0.001 0.002 0.003 0.005 0.007`, `--d 3 5 7 9`, `--batch-size 1024`.
- A missing default checkpoint gives a warning and that decoder is left out; a missing explicit path is an error.
- Prints three tables (logical error per shot, per round, decoding time per shot).
- Output in `eval_results/<timestamp>/`: `results.csv`, `results.json`, `eps_vs_p.png`, `decode_time_vs_d.png`.
- Run it on an idle machine; the timings depend on load, batch size, device and thread count.

## Where things end up

| Stage | Command          | Output                              |
| ----- | ---------------- | ----------------------------------- |
| 4, 7  | `train.py`       | `<out_dir>/` (`runs/...`)           |
| 5     | `experiment.py`  | `experiments/<timestamp>/`          |
| 6     | `select_best.py` | `best_hyperparameters/<timestamp>/` |
| 8     | `eval.py`        | `eval_results/<timestamp>/`         |
