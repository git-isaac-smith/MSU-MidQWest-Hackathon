# Million-shot bath timing controls

The eight timing policies use the five frozen layouts, distance-7 Z memory, seven syndrome rounds, one rail attachment, zero direct rail dephasing, and the native check, reset and measurement error probabilities of 0.001. Each policy evaluates 200,000 fresh shots per layout. The pooled binomial intervals describe Monte Carlo precision conditional on the frozen layouts. [Original timing controls](../bath-clock-controls-02/README.md)

| Policy | Failures in 1,000,000 shots | Logical error rate |
| --- | ---: | ---: |
| Fixed clock | 170,139 | 17.0139% |
| Global clock, independent calibration | 100,358 | 10.0358% |
| Global clock, installed calibration | 95,162 | 9.5162% |
| Shuffled clocks | 97,023 | 9.7023% |
| Individual fitted timing | 152 | 0.0152% |
| Full FFT timing with fitted frame and prior | 152 | 0.0152% |
| Dominant-tone timing with fitted frame and prior | 20,694 | 2.0694% |
| FFT-only timing, frame and prior | 152 | 0.0152% |

The fixed clock and individual fitted controls reference the corresponding native-background samples and scores after verifying the layout, full-precision physical circuit and full-precision prior circuit. Full FFT timing resolves to the same site assignments, physical circuit and prior circuit as individual fitted timing on all five layouts. Its result references the same scored records. [Bath background run](../million-bath-03/)

The full-precision physical gate arguments for FFT-only differ from the individual fitted circuit by at most $3.33\times 10^{-16}$ due to numerical differences in the frame calculation. FFT-only uses the fitted run's Stim seed and chunk size for each batch. Detector and observable array hashes matched in all 100 batches. Both priors produced identical predictions on those records, with 152 shared failures and no discordant outcomes. [Batch records and confidence intervals](result.json)

The independent calibration global clock, installed calibration global clock, shuffled clocks and the dominant-tone schedule each use distinct Stim seeds. The runner scores with the compiled herald-conditioned matching engine and verifies its predictions against the author's decoder on 1,024 fresh shots per layout and independently scored policy. `manifest.json` records the frozen inputs and circuit hashes. `result.json` contains the pooled and per-layout results, with paths and hashes for the score records. [Matching engine](../million-engine-03/)

Inference timing uses one warmed batch of 512 fresh held-out shots per layout and five timed calls to the compiled decoder. The mean of the five layout medians is divided by 512 for time per shot and by seven for time per syndrome round. These are wall times under shared machine load. Stim sampling, decoder construction, warm-up and scoring run outside the timer. The timings include the native matching graph initialization inside each decode call. Full FFT timing reuses the individual fitted measurements because both circuits and priors are identical. [Raw inference timings](accelerated-timing.json)

## Reproduce

From this repository root, point `$upstream` at the pinned checkout, revision `94572516042297d6b6018fe76eb2f2b547e3c46f`. Install the package versions in the manifest and build the matching engine. Complete the native-background samples in the companion bath run before running FFT-only. A rerun resumes at the first absent chunk.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -B experiments/million-controls-03/run.py $upstream --arms global global_installed_calibration shuffled dominant_tone_timing
python -B experiments/million-controls-03/run.py $upstream --arms fft_only
python -B experiments/million-controls-03/run.py --aggregate-only
python -B experiments/million-controls-03/run.py $upstream --timing-only
```

If a different platform produces different paired detector or observable records, `--fft-independent` selects independent seeds for the entire FFT-only row. Apply the same flag to aggregation for that run.
