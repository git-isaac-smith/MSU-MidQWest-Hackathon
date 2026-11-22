# Scaling the heralded bath control

The two archived bath clocks were tested separately on ideal rotated Z memories at distances 3, 5, 7, 15 and 31. Each shot applies one effective bath channel per data site and then measures one round of ideal checks. Both decoders see the same public check parities on each shot. The heralded version also sees the full location flag vector. The blind version uses the marginal X/Y flip rate. The heralded version uses the conditional pass rate on unflagged edges and zero-weight edges at flagged sites. It does not receive the actual Pauli label. [Accuracy and runtime table](index.html)

At the bath-selected 8.08 check time, the distance-31 blind matcher failed on 294 of 1,024 shots and the heralded matcher failed on zero. All 294 disagreements were recoveries. The 95% Wilson upper bound for the heralded error is 0.374%. At the fixed 4.00 check time, blind matching failed on 518 of 1,024 shots and heralded matching on 509. The paired change was −0.88 percentage points, with an approximate 95% interval spanning zero. Both clock rows preserve paired sampling within the clock, and the clocks have separately sampled physical channels.

Inference timings use 128-shot chunks after warm-up. The blind and heralded calls alternate in seeded random order. They exclude initial graph setup, channel sampling and scoring. Heralded time includes building flag-conditioned matching graphs, which is part of deployed inference. A concurrent main scaling benchmark was active on the same multicore host. The results use the existing graph-rebuilding decoder and give descriptive runtime measurements for this implementation. [Accuracy and timing CSV](measured/accuracy-timing.csv)

The geometry uses Stim detecting regions immediately after the reset, corresponding to the archived bath fault location. The run and verifier check that every detector and logical mask entry agrees with the original per-site injected-fault method at d3 and d5. Samples, public syndrome bits, all flags, and archived predictions are regenerated and compared by the verifier. A shot archive stores public syndrome and flag bits, separate evaluator truth, and predictions. All arrays are little-endian packed across columns. [Manifest and seeds](measured/run.json) · [Original bath experiment](../../bath-confirmation-01/README.md)

From the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.venv/Scripts/python.exe experiments/qubit-scaling-01/heralds/run.py run --out experiments/qubit-scaling-01/heralds/replay
.venv/Scripts/python.exe experiments/qubit-scaling-01/heralds/run.py verify --out experiments/qubit-scaling-01/heralds/measured
```
