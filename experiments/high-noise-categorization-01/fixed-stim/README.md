# Fixed Stim noise sweep

A distance-7, seven-round rotated surface-code memory was sampled in each basis at five physical noise settings. The provisional `simple_uniform` circuit sets Stim's after-Clifford depolarization, reset flip, measurement flip, and before-round data depolarization parameters to the same value `p`. Ordinary and correlated PyMatching used the same decomposed detector error model for each circuit and decoded the same 20,000 shots per scenario.

| Physical `p` | Basis | Ordinary logical errors | Correlated logical errors | Correlation wins / losses | Mean detectors firing |
|---:|:---:|---:|---:|---:|---:|
| 0.5% | X | 227 (1.14%) | 135 (0.68%) | 147 / 55 | 7.52% |
| 0.5% | Z | 214 (1.07%) | 141 (0.71%) | 125 / 52 | 7.45% |
| 1% | X | 2,472 (12.36%) | 1,935 (9.68%) | 1,172 / 635 | 13.90% |
| 1% | Z | 2,093 (10.47%) | 1,673 (8.37%) | 952 / 532 | 13.73% |
| 1.5% | X | 6,083 (30.42%) | 5,576 (27.88%) | 2,372 / 1,865 | 19.33% |
| 1.5% | Z | 5,270 (26.35%) | 4,866 (24.33%) | 1,913 / 1,509 | 19.10% |
| 2% | X | 8,653 (43.27%) | 8,388 (41.94%) | 3,024 / 2,759 | 23.90% |
| 2% | Z | 8,097 (40.49%) | 7,948 (39.74%) | 2,642 / 2,493 | 23.66% |
| 4% | X | 10,163 (50.82%) | 9,897 (49.49%) | 3,860 / 3,594 | 36.29% |
| 4% | Z | 10,050 (50.25%) | 10,067 (50.34%) | 3,256 / 3,273 | 35.97% |

The 1–1.5% range offers the strongest net gain in this sweep, 2.02–2.69 percentage points across X and Z. At 1.5%, the methods disagree on 17.11% of Z shots and 21.19% of X shots, with ordinary logical error rates still below 31%. At 2%, the ordinary error rate is above 40% and the net gain falls to 0.75–1.33 points. At 4%, both methods are near 50% logical error. Correlated decoding still gains 1.33 points in X (paired approximate 95% CI 0.48–2.18 points), while the Z difference is uncertain (paired approximate 95% CI for the gain −0.88 to +0.71 points).

Detector firing fraction is the number of active detectors divided by all 336 detectors in the circuit. At 1.5%, the 10% to <25% bin contains over 96% of shots in both bases. In the ≥25% bin, correlation has 96 wins and 131 losses in X (680 shots) and 95 wins and 90 losses in Z (588 shots). These exploratory bins describe observed syndromes under the specified circuit and seed. The counts support comparisons within each physical noise setting. Density alone cannot identify the underlying physical fault location or mechanism.

- [scenarios.csv](scenarios.csv) has the raw counts, per-decoder Wilson 95% intervals, paired approximate 95% intervals for correlated minus ordinary logical error rate, and per-batch inference timings. A correlation win is a shot on which only ordinary PyMatching fails.
- [density.csv](density.csv) repeats the paired scoring for every nonempty density bin. The paired approximation requires at least 30 discordant shots.
- [manifest.json](manifest.json) records the sampling and timing protocol and package versions. Timings are single-pass wall-clock measurements of five 4,000-shot calls per decoder, after a warm-up, excluding setup, sampling, and scoring.

Run from the repository root with the project dependencies installed.

```powershell
.venv/Scripts/python experiments/high-noise-categorization-01/fixed-stim/run.py
```
