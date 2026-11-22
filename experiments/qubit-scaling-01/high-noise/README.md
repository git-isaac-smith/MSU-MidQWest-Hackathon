# Higher-noise decoder stress test

This secondary workload raises the physical error parameter to 0.01 for the provisional circuit-level Stim model. The rotated Z-memory circuits use distances 7, 15 and 31 with rounds equal to distance. Within each case, all decoders read the same 2,048 public detector records. Logical observables stay with the evaluator. An independent seed namespace separates these records and the CSS cutoff calibration from the primary benchmark. [Accuracy and timing table](index.html)

At distance 31, ordinary MWPM failed on 801 of 2,048 shots (39.111%), native correlated MWPM on 444 (21.680%), and the bath-clock correlated decoder on 438 (21.387%). The equal-axis correlated control failed on 445 (21.729%). Relative to that control, the bath-clock decoder recovered 179 failures and regressed on 172, giving two-sided exact McNemar p = 0.749.

The protocol fixed the six decoders before sampling. Ordinary and native correlated MWPM use the native detector error model. The equal-axis and bath-clock correlated models use the same expansion of the idle channel. Both retain the simulated total idle error probability. The weights use an equal mixture or the saved bath-clock Pauli ratios, respectively. Library weighted Union-Find uses the native graphlike decomposition. CSS activity routing uses ordinary matching on observable-bearing components and escalates high-activity shots to native correlated matching. Its cutoff is the 94th percentile of counts on 1,024 independent unlabeled records. [Frozen protocol](measured/protocol.json)

Inference time is the median of warm calls over 128-shot chunks. Each decoder runs in its own sequential worker with BLAS and OpenMP limits of one. Calls begin with packed rows and include any method-required unpacking. Setup, Stim sampling, and scoring take place outside the timer. The study ran while the main benchmark was also running on the Intel Core i7-13700HX host. Absolute timing depends on host load and batch size.

The CSV records logical error, Wilson intervals, and recoveries and regressions against both ordinary MWPM and expanded equal-axis correlated matching. The paired change intervals are conservative 95% intervals obtained from simultaneous exact binomial intervals for the frequency of disagreements and their direction. McNemar tests use the same-shot disagreements. The verifier regenerates every detector and observable bit, checks circuit, sample and prediction hashes, and independently recounts the results. [Metrics CSV](measured/metrics.csv)

From the repository root:

```powershell
.venv/Scripts/python.exe experiments/qubit-scaling-01/high-noise/run.py run --out experiments/qubit-scaling-01/high-noise/replay
.venv/Scripts/python.exe experiments/qubit-scaling-01/high-noise/run.py verify --out experiments/qubit-scaling-01/high-noise/measured
.venv/Scripts/python.exe experiments/qubit-scaling-01/high-noise/report.py --metrics experiments/qubit-scaling-01/high-noise/replay/metrics.csv --out experiments/qubit-scaling-01/high-noise/replay.html
```

