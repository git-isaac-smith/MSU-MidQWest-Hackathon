# Distance-7 bath timing with raised background check errors

At 2% added check error per Pauli axis, the fitted balanced check failed on 3.280% of shots versus 37.185% for the frame-calibrated clock. The fitted arm had the lower logical error rate at all five tested levels. The clock-to-fitted rate ratio fell from about 829 at the native level, where the fitted arm had four failures, to about 3 at the highest level.

| Added X error and added Z error, each | Frame-calibrated clock failures | Clock rate and 95% Wilson interval | Fitted balanced-check failures | Fitted rate and 95% Wilson interval |
| ---: | ---: | ---: | ---: | ---: |
| 0.1% | 3,317 / 20,000 | 16.585% [16.076%, 17.107%] | 4 / 20,000 | 0.020% [0.008%, 0.051%] |
| 0.5% | 4,446 / 20,000 | 22.230% [21.659%, 22.811%] | 35 / 20,000 | 0.175% [0.126%, 0.243%] |
| 1% | 5,563 / 20,000 | 27.815% [27.198%, 28.440%] | 136 / 20,000 | 0.680% [0.575%, 0.804%] |
| 2% | 7,437 / 20,000 | 37.185% [36.518%, 37.857%] | 656 / 20,000 | 3.280% [3.042%, 3.536%] |
| 4% | 9,261 / 20,000 | 46.305% [45.615%, 46.997%] | 3,098 / 20,000 | 15.490% [14.995%, 15.998%] |

The pooled Wilson intervals describe Monte Carlo precision for the five fixed layouts. Each layout contributed 4,000 shots to every arm and error level. [Counts, intervals, layouts, and seed provenance](result.json)

## Intervention

This sweep extends the [Table 5 replay](../sakidja-table5-replay-01/README.md) in distance-7 rotated **Z memory** with seven syndrome rounds, the first rail attachment, zero direct rail dephasing, and clock period 6. We vary only the upstream `P_CHECK` setting. Each round, each data qubit receives independent `X_ERROR` and `Z_ERROR` operations whose probabilities include this common additive term. The sampled circuit and decoder-prior circuit each receive the new value. In Z memory, data X errors can directly flip the stored logical Z value. Syndrome readout and reset each retain the upstream 0.1% flip probability. [Pinned upstream circuit and decoder](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py#L389-L510)

The decoder uses the true bath channel for the clock and the fitted bath channel for the balanced check. The saved check times, physical bath channels, decoder bath priors, and five published layouts stay fixed across the sweep. Across the 49 unseen baths, the mean physical erasure probability is 31.62% for the clock and 7.73% for the fitted check. The clock checks at time 6 and the fitted check averages 11.17. [Published per-bath channels](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/raw_round_channels_per_bath.csv)

The average herald count stayed between 107.48 and 107.62 per clock shot and between 24.94 and 25.12 per fitted-check shot. Over the same range, the average number of non-herald detector events rose from 121.26 to 129.14 for the clock and from 39.22 to 67.48 for the fitted check. Erasures can also trigger syndrome detectors, so these averages summarize the observed records rather than counting latent Pauli faults. The JSON includes detector totals by layout and logical outcome. [Detector counts](result.json)

## Reproduce

The driver uses the same saved physical and fitted channels as the Table 5 replay, with 50 distinct fresh Stim seeds across the ten points and five layouts. A same-seed check at the native setting exactly reproduced the upstream logical-failure counts for 500 shots per arm. Dependency versions are in the result file. [Versions and seeds](result.json)

Run from the repository root in PowerShell.

```powershell
$upstream = Join-Path $env:TEMP 'sakidja-rql-quantum-bath'
git clone https://github.com/sakidja/RQL_Quantum_bath.git $upstream
git -C $upstream checkout 94572516042297d6b6018fe76eb2f2b547e3c46f
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python experiments/bath-background-stress-01/run.py $upstream
```
