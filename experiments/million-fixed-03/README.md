# One-million-shot fixed-circuit comparisons

Each failure is one wrong logical observable on one complete rotated surface-code memory shot. Stim supplies the evaluator with a detector record and a separately sampled logical observable. The decoder receives the detector record. All runs use the project's provisional `simple_uniform` circuit model and as many measurement rounds as the code distance.

| Study | Settings | Scored denominator |
| --- | --- | --- |
| Fixed-noise grid | X/Z memory, distances 3, 5 and 7, noise of 0.1%, 0.3%, 0.5% and 1%. Both bases at distance 7 also run at 1.5%, 2% and 4%. | 1,000,000 per circuit and decoder |
| Distance policy and bath priors | Six policies over the original 24 distance, basis and noise strata. The basis-selected prior uses the clock prior for X and equal-axis weights for Z. | 1,000,000 per pooled policy, exactly 41,666 or 41,667 from each stratum |
| Physical fault removal | Distance-7 Z memory at 0.5%, 1% and 2%, with correlated matching. Compare all faults with each of four paired removals. | 1,000,000 complete records per noise level, scored five ways |
| Large codes at 1% | Six published decoders on distance-7, 15 and 31 Z memory | 1,000,000 per decoder and distance |
| Large codes at 0.5% | Four primary scaling methods covering ordinary, native correlated, equal-axis correlated and bath-clock correlated matching. Distances 15 and 31, both bases. | 1,000,000 per decoder and circuit |

The [protocol](protocol.json) contains the exact grid, seed algorithm, software versions and source hashes. Chunks carry the seed, sample hash, counts and decoding time. These timings measure the batch call while other jobs share the host. In fault-removal comparisons a shot can improve under more than one removal. The counts for different removals therefore overlap.

## Replay

From the repository root, use the Python environment from the quick start. Install the campaign and PDF dependencies with `python -m pip install -e ".[scaling,hypergraph,reports]"`. These commands resume the existing checkpoints. Use a separate checkout for an independent replay. Start a single scaling coordinator for a given results directory.

```sh
python experiments/million-fixed-03/run.py grid --workers 3
python experiments/million-fixed-03/run.py fault --workers 2
python experiments/million-fixed-03/run.py pooled --workers 2
python experiments/million-fixed-03/large_queue.py --workers 16
python experiments/million-fixed-03/audit.py --require-complete --replay-lookup
```

The scaling coordinator distributes disjoint seed ranges and merges them by index. Every decoder at a large-code setting samples the exact same Stim batches. The audit compares their sample hashes, verifies failure masks and emits matched recoveries and regressions in `large-paired.csv`. The lookup replay regenerates every distance-3 Z-memory record at 0.5% and scores the archived portable lookup independently.

The [results PDF](../../output/pdf/Decoder%20error%20rates%20-%20one%20million%20shots.pdf) and [results CSV](combined-results.csv) include the [bath simulation](../million-bath-03/README.md) and [timing controls](../million-controls-03/README.md). A progress copy prints actual completed counts. The final export checks that all planned denominators were reached. With ReportLab installed in the rendering environment, run the following from the repository root.

```sh
python scripts/build_million_error_sheet.py --require-complete
```

The [one-page fixed-noise summary](../../output/pdf/Decoder%20error%20rates%20-%20one%20million%20fixed-noise%20shots.pdf) compares the three code distances at 0.5% noise in Z memory and the six policies across 24 settings. Rebuild it from the completed grid, pooled, and fault-removal checkpoints with `python scripts/build_million_fixed_summary.py`.
