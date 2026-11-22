# How decoder performance changes with qubit count

Open the [logical error and decoder runtime page](index.html) to compare any circuit-level decoder with ordinary matching. Each run stores one logical qubit. Code distances 3, 5, 7, 15 and 31 use 9, 25, 49, 225 and 961 **data** qubits. The largest circuit uses another 960 measurement ancillas and measures 31 rounds. The independent qubit IDs used by Stim have gaps, so the physical-qubit count comes from the active circuit coordinates.

## Where the choice of decoder changes

At distance 3 and 0.5% physical noise, the portable posterior lookup made 136 errors in 10,000 Z-memory shots at 0.12 µs/shot. Ordinary matching made 154 errors at 0.46 µs/shot. The table grows with the number of possible syndromes. Distance 5 already has 120 detector bits, beyond the implementation's 24-bit limit.

At distance 15 and the same noise level, projected ordinary matching made 12 errors in 10,000 Z-memory shots at 75.29 µs/shot. Ordinary matching made 12 at 158.03 µs/shot. Native correlated matching made 2 at 455.42 µs/shot. The projected method focuses ordinary matching on detector components connected to the logical observable. The correlated method spends more time modeling which detection events may share a fault. [Circuit-level measurements](metrics.csv)

At distance 31 (1,921 physical qubits), ordinary matching made zero X-memory errors and one Z-memory error in 10,000 shots per basis. Native correlated, equal-axis correlated and fixed-clock bath-weighted correlated matching each made zero in both bases. A zero count per basis has a 95% upper Wilson limit of 0.0384%. This sample separates their computational costs more clearly than their accuracy. Ordinary matching took 2.22 ms/shot in X and 2.52 in Z. Native correlated matching took 7.03 and 7.94, respectively. On the 1,024-shot pilots, projected matching took 1.24 and 1.04 ms/shot with zero observed errors in each basis.

The hard temporal filter discarded too much syndrome information. On identical 128-shot distance-31 prefixes, it made 65 errors in X memory and 56 in Z. Local-pair peeling made zero in both prefixes. Its batch times of 5.49 ms/shot in X and 5.41 in Z exceeded ordinary matching at this size. The filtered records and the untouched records use the same physical samples.

The unrestricted logical-gap router exceeded 180 seconds on its planned 1,024-shot run in each basis. Follow-ups on the first 16 records took 430 ms/shot in X and 339 ms/shot in Z. The capped logical-gap implementation completed its full 1,024-shot pilot in each basis. Both exact lookup interfaces and the sparse posterior cache reached their size or memory limits at distance 31. The other 32 configurations produced predictions there, including the short logical-gap probes. [Metrics and limits](metrics.csv)

## What was compared

All 35 fixed-circuit configurations use the same detector order and the same test shots within a case. The baseline is ordinary minimum-weight perfect matching in PyMatching. Every prediction is a logical flip, and each shot counts as one failure if the prediction differs from Stim's sampled observable. Physical noise is the provisional `simple_uniform` model with parameter 0.005. A [second study at 0.01](high-noise/index.html) compares six methods in Z memory. It uses a separate seed and shot count.

The main experiment generated 10,000 shots for each distance and memory basis. The predeclared prefixes for computationally expensive methods range from 16 to 4,096 shots. Use the paired counts against the baseline on the same prefix. Zero failures in 16 shots still permits a logical error rate as high as 19.36% at the upper end of a two-sided Wilson 95% interval. For 10,000 shots the corresponding upper bound is 0.0384%. The [protocol](protocol.json) records every planned prefix.

**Runtime** is the median elapsed time for a full, warmed inference batch divided by its shot count. Each call starts with a boolean array of complete detector records in host memory. The timer includes any conversion to packed data required by that implementation. The standard batch contains 128 records. The smallest pilot contains 16. The measurements exclude generating the Stim records, computing error statistics and building the decoder. These are CPU measurements of buffered records. The machine also ran some of the separate workload comparisons during the sweep, so small timing differences merit another measurement on the judging hardware.

**Training** uses an independent seed. The learned router loads its archived policy for matching distance-5 and distance-7 Z circuits. Other geometries get a new calibration with a smaller training budget. Activity-count bins can saturate as the detector volume grows, and the learned router disables some spatial or temporal features beyond its supported lane counts. The two exact-posterior interfaces accept at most 24 detector bits. The distance-31 sparse posterior cache would need 3.39 GiB for one fault-pair array before allocating the rest of its tables. The study records unsupported or resource-limited cells explicitly.

## FFT/CNN and heralded bath results

- **FFT/CNN, five-round Z memory.** At distance 33, direct FFT site parity made 462 errors in 1,024 shots (45.12%). Adding terminal spatial matching gave zero errors on the same shots, as did ordinary MWPM. The FFT hybrid took 2,142.61 µs/shot versus 88.67 µs/shot for ordinary MWPM. Both zero-error estimates have a 0.374% upper Wilson bound. These checkpoints were trained at distance 5. The five-round noise model and ideal terminal readout differ from the main circuit. [Full FFT and runtime comparison](fft/index.html)
- **Bath with measured location flags.** At distance 31 and the bath-selected model time 8.08, the flag-aware decoder made zero errors in 1,024 shots. Hiding the flags gave 294 errors (28.71%). Reading and using the flags cost 2,652.08 µs/shot versus 139.02 µs/shot for blind matching. At the fixed model time 4.00, the two decoders made 509 and 518 errors, respectively. Each within-clock comparison isolates the use of flags. Changing the clock changes the physical experiment. This simulator applies one effective channel per site with ideal checks. The main benchmark uses repeated noisy measurements. [Full heralded bath comparison](heralds/index.html)

## How the methods behaved at 1% physical noise

Larger codes had more failures per complete memory at this higher noise level. Native correlated matching went from 9.91% at distance 7 to 11.67% at 15 and 21.68% at 31. Ordinary matching rose from 11.62% to 18.55% and 39.11%. Increasing distance also increases the number of measurement rounds in this workload. The data show finite-size performance at two noise settings. Establishing a threshold would require a finer noise sweep.

At distance 31, correlated matching corrected 587 ordinary-matching failures and introduced 230 others on the same 2,048 Z-memory shots. The net improvement was 17.43 percentage points. The fixed-clock bath weights gave the lowest observed error among the six methods in this stress test. Against the equal-axis correlated control they corrected 179 mistakes and introduced 172, a seven-shot difference with exact paired McNemar p = 0.749. These data support the benefit of correlated matching on this case. They leave the incremental value of the bath weights unresolved. [Stress-test scores, intervals and paired comparisons](high-noise/measured/metrics.csv)

| Decoder | Logical failures / 2,048 | Logical error | Batch ms/shot |
| --- | ---: | ---: | ---: |
| Ordinary matching | 801 | 39.11% | 10.30 |
| Native correlated matching | 444 | 21.68% | 26.37 |
| Equal-axis correlated matching | 445 | 21.73% | 27.09 |
| Fixed-clock bath-weighted correlated matching | 438 | 21.39% | 27.59 |
| Weighted Union-Find | 868 | 42.38% | 19.82 |
| Activity routing | 780 | 38.09% | 7.43 |

## Replay

Install the project with its `hypergraph` and `scaling` extras. Run these commands from the repository root. Data preparation requires a new output location. The run command can resume from its completed measurement files.

```sh
python -m bascd.scale_study prepare --out results/my-scaling
python -m bascd.scale_study run --out results/my-scaling
python -m bascd.scale_study short-probes --out results/my-scaling
python experiments/qubit-scaling-01/audit.py --run results/my-scaling --paired-out results/my-scaling/d31-paired.csv
python -m bascd.scale_report results/my-scaling
```

The run folders contain the exact Stim circuits, packed detector events and separate observable labels, saved prediction arrays, per-batch timing samples, calibration seeds and graph settings. The audit regenerates the held-out records and recounts each measured method. The short-probe command retains the original time-limit record and attempts the first 16 shared shots for eligible methods. Each linked companion study has its own verifier and replay instructions. [Software versions and implementation fingerprints](implementation.json) · [Distance-31 common-prefix comparisons](d31-paired.csv)
