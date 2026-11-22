# qecdec: hybrid neural decoder for the rotated surface code

Predicts the logical flip of a rotated-surface-code Z-memory experiment (`rounds = d`, circuit-level
noise) from detection events, and compares it with MWPM (PyMatching) and a dense 3D-CNN baseline.
All training data is sampled on the fly from Stim; nothing is stored, there are no epochs.

```
.
├── README.md
├── requirements.txt
├── pytest.ini
├── configs/            hybrid.yaml  cnn_baseline.yaml  smoke_d3.yaml
├── qecdec/
│   ├── config.py       flat Config dataclass, YAML + key=value overrides
│   ├── utils.py        seeds, device, eps_round, Wilson interval
│   ├── engine.py       TrainData stream, val sets, predict, optimiser, scheduler, train_step
│   ├── data/
│   │   ├── stim_source.py  circuit, seeded ShotSource, PyMatching MWPM
│   │   ├── geometry.py     detector coords -> grid, type/weight, boundary channels, GeometryCache
│   │   └── batch.py        Batch dataclass, build_batch (dense grid + padded event tokens)
│   └── models/
│       ├── maskconv.py     mask-aware 3D conv (renormalise by valid taps)
│       ├── attention.py    Fourier features, learned relative bias, transformer block
│       ├── hybrid.py       stem -> tokens (+4 boundary, +1 global) -> attention -> logit
│       └── cnn_baseline.py dilated residual masked CNN -> mean+max pool -> logit
├── train.py  eval.py  benchmark.py
└── tests/   helpers.py  test_geometry.py  test_data.py  test_models.py  test_smoke.py
```

## Design and math

**Coordinates -> grid** (`data/geometry.py`). Stim places data qubits at odd-odd `(x, y)` and
stabiliser ancillas (hence detectors) at even-even `(x, y)`, `0 <= x, y <= 2d`; `t = 0..d`.

$$g_x = x/2,\quad g_y = y/2,\quad g_t = t \;\Rightarrow\; \text{grid } [T,H,W] = [d+1,d+1,d+1].$$

Type and weight (verified against the circuit's gate structure in tests):

$$
\text{type}=\begin{cases}X & g_x+g_y \text{ odd}\\ Z & \text{else}\end{cases},\qquad
\text{weight}=\begin{cases}2 & g_x\in\{0,d\}\ \text{or}\ g_y\in\{0,d\}\\ 4 & \text{else}\end{cases}
$$

In a Z-memory the first and last detector layers contain only Z checks, so the validity mask
depends on `t` too. Grid channels (`C_IN = 10`): event, mask, clipped distance to
left/right/bottom/top (`min(dist, C)/C`), type (+1 X / -1 Z), weight/4, first-round flag,
last-round flag. Static channels are multiplied by the mask. Everything is defined by distance to a
boundary or round index, so it is valid at any `d`.

**Mask-aware conv** (`maskconv.py`).

$$y(v)=b+\frac{K^3}{n(v)}\sum_{u\in\mathcal N(v)} m(u)\,W[u-v]\,x(u),\qquad n(v)=\sum_{u\in\mathcal N(v)} m(u),$$

with the output multiplied by `m(v)`. A boundary site then has the same output scale as an interior
one (tested: all-ones input gives exactly `K^3` everywhere valid). No reflect padding.

**Hybrid model** (`hybrid.py`). Stem (2 mask-aware convs, GroupNorm, GELU, width 64) -> gather stem
features at fired detectors -> concat Fourier features
$[\sin(w_k c),\cos(w_k c)]$, $w_k=\pi 2^k/16$ and the 8 boundary features -> linear to 64 ->
sequence `[global | 4 boundary | events...]`, right-padded with a key-padding mask -> 3 pre-LN
transformer layers, 4 heads, logits

$$
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+b\right)V,\quad
b_{ij}=\mathrm{MLP}\big(\mathrm{clip}(\Delta x),\mathrm{clip}(\Delta y),\mathrm{clip}(\Delta t)\big).
$$

Event<->boundary-token bias is a learned table indexed by the event's clipped distance to that
boundary; the rest is one learned scalar per head; padded keys get `-inf`. Because the clipped
offsets take only $(2C+1)^3$ values, the MLP is evaluated once on that table and gathered. The
global token is never padded, so a shot with zero events still works. Readout: LayerNorm on the
global token -> linear -> one logit, BCE loss.

**CNN baseline** (`cnn_baseline.py`). Input mask-aware conv, then dilated residual blocks. Receptive
field per axis

$$\mathrm{RF}=1+(k-1)\Big(1+\sum_i \delta_i\Big)$$

is asserted `>= d` (default dilations `[1,2,4,1]` give 19). Masked mean + max pooling -> MLP -> logit.

**Statistics** (`utils.py`).

$$\epsilon=\tfrac12\Big(1-(1-2P_L)^{1/\text{rounds}}\Big),\quad \text{rounds}=d.$$

Error bars are 95% Wilson intervals on `P_L`; the endpoints are pushed through the monotone map to
get per-round bars. All decoders in `eval.py` see the same shots in each (d, p) cell.

**Important baseline note.** "Beat random" (LER < 0.5) is a very weak bar: at d=3, p=0.005 the
undecoded flip rate is ~0.10 and MWPM gets ~0.017. The smoke test therefore requires the model
to be below _half the trivial "never flip" rate_.

## Commands (run from this directory, in order)

```
pip install -r requirements.txt
pytest -q -m "not smoke"                         # 64 tests, a few seconds
pytest -q -m smoke                               # d=3 training, ~3 min on CPU, faster on GPU
python train.py --config configs/hybrid.yaml
python train.py --config configs/cnn_baseline.yaml
python benchmark.py --config configs/hybrid.yaml
python eval.py --hybrid runs/hybrid/best.pt --cnn runs/cnn_baseline/best.pt
```

Expected output:

1. `pytest -m "not smoke"`: `64 passed`.
2. smoke: `1 passed`; the log shows val LER falling from ~0.10 (trivial) to ~0.015 (about MWPM level, 0.017).
3. `train.py`: first line `[train] model=hybrid params=... device=cuda ...`, then `[train] step ... loss ... ler ... samples/s`
   every 100 steps and `[val] step ... d3=.. d5=.. d7=.. dmean=..` every 1000 with `new best ... -> best.pt`.
   Files: `runs/<name>/{config.json,log.jsonl,last.pt,best.pt}`. Train LER should drop quickly from ~0.15 and
   approach MWPM level on val.
4. `benchmark.py`: one line per distance (ms/step, samples/s, hours for 1e7 and 1e8 samples) plus a
   uniform-mixture line.
5. `eval.py`: three tables (per shot, per round, decoding time per shot) with rows for each `(d, p)` and
   columns MWPM / CNN / Hybrid;
   `*` marks d outside the training set (d=9). Don't expect the hybrid to match MWPM at low `p`
   (trained at p=0.005); how close it gets is the actual result of the study. It also saves
   `results.csv`, `results.json`, `eps_vs_p.png` and `decode_time_vs_d.png` to `eval_results/<YYYYmmdd_HHMMSS>/`; each run gets
   a new folder (a `_1`, `_2`, ... suffix if the same second is reused), so nothing is overwritten.
   Change the root with `--out-dir`.

Ablations: `--set use_boundary_channels=false use_boundary_tokens=false mask_aware_conv=false`
(use a different `out_dir=`).

## Assumptions to verify

- Stim 1.16 keyword is `after_clifford_depolarization` (not `...depolarizing`); `generated()` takes
  keyword-only `distance` / `rounds`. Verified on 1.16. Noise: all four parameters set to `p`.
- Detector coordinates are `(x, y, t)` with even `x, y` in `[0, 2d]` and integer `t` in `[0, d]`
  (`build_numpy_geometry` raises otherwise). One detector per grid cell; asserted.
- Stabiliser type/weight rules are checked against Hadamard and CX counts in the circuit for d = 3, 5, 7, 9
  but only for `rotated_memory_z`.
- PyMatching 2.x: `Matching.from_detector_error_model(dem)` and `decode_batch(det)` returning `[N, 1]`.
- `compile_detector_sampler(seed=...)` and `sample(n, separate_observables=True)` return bool arrays.
- The "boundary type" channel is encoded as the stabiliser type (+1 X / -1 Z). Which boundary is rough or smooth
  follows from the fixed channel order (left/right carry Z weight-2 checks, bottom/top X).
- GroupNorm in the stem/blocks averages over invalid (zeroed) sites too; the output is re-masked afterwards.
  A masked norm would be cleaner but I did not add it.
- Only odd `d >= 3` are supported; the hybrid is distance-agnostic (Fourier + relative bias); the CNN is too
  but needs RF >= max d.
- Performance numbers from this machine were CPU only; CUDA, `bf16`, `channels_last`, `torch.compile`
  and SDPA with a float mask were not run on a GPU here. The `channels_last_3d` path was only tested
  for layout, not speed.
- `train.py` prints the mean key as `dmean`; cosmetic.
