# Bath timing controls for distance-7 Z memory

This experiment tests whether a check time needs to follow its local bath model to retain the logical-error reduction seen in [the Table 5 replay](../sakidja-table5-replay-01/README.md). It uses the same five published bath layouts, seven syndrome rounds, 4,000 fresh shots per layout and arm, the first rail attachment, zero direct rail dephasing, and `P_CHECK = P_MEAS = 0.001`. Each arm has its own Stim seeds. The circuits and herald-conditioned matching decoder come from the pinned author source. [Upstream model, circuit, and decoder](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py)

## Results

| Schedule | Mean time | Mean physical erasure per site | Logical failures / 20,000 | Logical error rate |
| --- | ---: | ---: | ---: | ---: |
| Fixed 6 | 6.00 | 31.36% | 3,419 | 17.095% |
| Global, independent calibration | 15.00 | 27.77% | 1,983 | 9.915% |
| Global, installed calibration | 14.88 | 27.43% | 1,929 | 9.645% |
| Shuffled | 11.10 | 27.43% | 1,892 | 9.460% |
| Individual fitted | 11.10 | 7.30% | 4 | 0.020% |
| Full FFT timing, fitted frame and prior | 11.10 | 7.30% | 2 | 0.010% |
| Dominant-tone timing, fitted frame and prior | 10.60 | 20.49% | 396 | 1.980% |
| FFT-only timing, frame and prior | 11.10 | 7.30% | 6 | 0.030% |

Shuffling raised the logical error rate from 0.020% to 9.460% while preserving the time histogram of every layout. The two schedules share 41 distinct times across the 245 site occurrences, range from 6.36 to 15.00, and have quartiles of 8.94, 11.40 and 13.08. The permutation changed the time assigned to 236 site occurrences. [Per-layout control results and timing distributions](result.json)

Full FFT timing reproduced every fitted time and yielded identical physical and prior circuit fingerprints when the replay retained the fitted frames and priors. With FFT-derived frames and priors, the largest relative frame error against the direct model was $6.72\times10^{-9}$ radians, and the largest absolute erasure and conditional-Z errors were $8.77\times10^{-9}$ and $1.81\times10^{-9}$. All five fitted and FFT matching graphs shared their edge support, fault IDs and herald-conditioning map. The largest absolute matching-weight difference was $1.24\times10^{-7}$. The small failure counts for these three arms come from independent shots. [Spectrum replay results and graph verification](fft_result.json)

## Schedule selection

Every optimized schedule uses the author's 151-point grid from 6 to 15 in steps of 0.06. The balanced objective is $p_X+p_Y+p_Z+p_\mathrm{erase}/8$, with the Pauli probabilities conditional on the rail check passing. Selection uses calibration before circuit simulation. [Frozen calibration and seed plan](calibration.json)

- **Fixed 6.** Reconstructs the published fixed clock with the author's true-parameter frame and matching prior.
- **Global, independent calibration.** Minimizes the mean balanced cost over the 400 training baths with saved fits. All evaluation sites use the resulting single time. The training bath IDs are disjoint from the 49 held-out baths used to construct the evaluation layouts.
- **Global, installed calibration.** Minimizes the same cost over the fitted models of the 245 installed sites across the five layouts, with equal weight per site occurrence. All evaluation sites use one common time. This gives the global clock access to the same local calibration models as the individualized schedule.
- **Shuffled.** Randomly permutes the individualized schedule within each layout and preserves its exact multiset of check times. The permutation seed is fixed per layout before sampling.
- **Individual.** Minimizes each bath's fitted balanced cost. The reconstructed assignments and physical channels are checked against the published values for all 49 baths.

For both global arms, shuffled and individual, each site gets a frame and matching prior recomputed from its own fit at the assigned time. The physical channels use the published true parameters under those frames. The runner writes the schedules and seeds before reading any published physical channels or generating evaluation shots. [Published per-bath parameters](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/raw_parameters_per_bath.csv) · [Published physical channels](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/raw_round_channels_per_bath.csv)

## Spectrum-based schedules

The spectrum replay uses the same layouts and independently generated shots. The FFT timing and dominant tone timing arms keep the local fitted frames and priors at their assigned times, isolating the effect of timing. The FFT-only arm takes each bath's time, frame phase, erasure rate and conditional Pauli-Z rate from simulated calibration traces. Its frame is `diag(1, exp(i*phase))`, matching the author's convention that `round_channel` applies the inverse of the supplied frame. The replay verifies the relative phase and predicted probabilities against direct propagation and compares the fitted and FFT matching graphs. [Calibration trace reconstruction](../bath-fft-timing-02/) · [Frozen spectrum inputs](fft_calibration.json)

The calibration traces are noiseless and cover 5,000 samples through time 1,500. The model maps each assigned delay to a local channel in each of seven rounds, starting each round in the bath ground state. The rates are measured per seven-round shot. Grouped schedules, noisy calibration and implementation of asynchronous checks remain open. [Upstream model assumptions](https://github.com/sakidja/RQL_Quantum_bath/tree/94572516042297d6b6018fe76eb2f2b547e3c46f#what-the-paper-does-not-cover)

## Decoder timing

For each of the 40 arm-layout batches, the timing runner reconstructs both circuits and verifies the full 4,000-shot sample against its archived fingerprint. It warms the compiled decoder once on the first 512 detector records and measures three serial `decode` calls on those same records, in independently shuffled batch order. Circuit and graph construction, sample generation, and comparisons run outside the timer. Each arm reports the mean of its five layout medians, divided by 512 for cost per shot and then by seven for cost per round. [Raw timing results and prefix fingerprints](timing.json)

## Run locally

From the repository root, use an upstream checkout at revision `94572516042297d6b6018fe76eb2f2b547e3c46f` and the Python package versions recorded in [result.json](result.json).

```powershell
$upstream = Join-Path $env:TEMP 'sakidja-rql-quantum-bath'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -B experiments/bath-clock-controls-02/run.py $upstream
python -B experiments/bath-clock-controls-02/fft_replay.py $upstream experiments/bath-fft-timing-02/per_bath.csv
python -B experiments/bath-clock-controls-02/timing.py $upstream
```
