# Iteration 01 · Surface-code benchmark

Both decoders recovered all 1,000 X-memory shots. On Z memory they disagreed once. Correlated matching recovered that shot. A single disagreement leaves the accuracy ranking open.

| Memory | Decoder          | Failed / shots | Logical error rate | 95% Wilson interval | Batch µs/shot |
|:-------|:-----------------|---------------:|-------------------:|--------------------:|--------------:|
| X      | MWPM             |      0 / 1,000 |              0.00% |       0.00%–0.38%   |          0.28 |
| X      | Correlated MWPM  |      0 / 1,000 |              0.00% |       0.00%–0.38%   |          0.71 |
| Z      | MWPM             |      1 / 1,000 |              0.10% |       0.02%–0.56%   |          0.22 |
| Z      | Correlated MWPM  |      0 / 1,000 |              0.00% |       0.00%–0.38%   |          0.61 |

Each decoder saw the same shots within each basis. The simulation used distance 3, three rounds, and `simple_uniform` noise at `p = 0.001`. The seeds were 101 for X and 102 for Z. Timing divides one local batch decode by 1,000 shots.

[Full run and paired counts](results/iteration-01/benchmark.json) · [Measurement rows](results/iteration-01/benchmark.csv)
