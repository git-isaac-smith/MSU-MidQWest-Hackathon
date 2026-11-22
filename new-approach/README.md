# Locating data-qubit errors from five rounds of surface-code checks

[Latest decoder optimization at distances 5 and 7, with held-out logical error percentages](POSTERIOR.md)

[Trained PyMatching and FFT logical hybrid, paired evaluation and deployment](HYBRID.md)

## Stim GitHub reference versus FFT

The Stim getting-started notebook decodes the simulated detector record with ordinary minimum-weight perfect matching (MWPM) in PyMatching. Its error percentage is **100 × wrong logical predictions / shots**. We ran that method on our distance-5 Z memory with five check rounds, 0.8% data X faults per qubit per round, 4% readout flips per ancilla per round, and ideal final data readout. These are the prototype's controlled noise settings. [Stim tutorial](https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb) · [PyMatching example](https://pymatching.readthedocs.io/en/stable/#decoding-stim-circuits)

| What is counted | Reference matching | FFT method |
| --- | ---: | ---: |
| Logical bit wrong after final readout, 100,000 fresh shots | 425 / 100,000 = **0.425%** | 2,023 / 100,000 = **2.023%** with final matching |
| Physical data site identified incorrectly after five rounds, 250,000 site decisions | 4,833 / 250,000 = **1.9332%** | 6,120 / 250,000 = **2.4480%** |

The **standalone FFT + CNN** predicts the data sites from the five noisy check rounds. Taking the logical parity of those calls directly gave **14.832%** logical error on the same 100,000 shots. For the final-readout comparison, we added one fixed spatial matching step to reconcile the FFT calls with the public final check parities. The reference uses those same final checks in its full matching graph. Stim returns the true logical bit separately for scoring. The FFT checkpoint and decision threshold come from the previous screen run.

The physical-site comparison reuses that run's 10,000 labeled test shots (25 sites per shot). Both methods receive exactly five noisy check grids. For this measure, matching tracks physical sites on a minimum-weight fault history. Different boundary qubits can produce identical check records, so even a valid correction can disagree with the actual site. The earlier raw-check CNN comparison below is a model-input experiment.

[0–100% error graph](results/screen-run-2026112102/stim-comparison/stim-vs-fft-error-percent.png) · [Counts and percentages](results/screen-run-2026112102/stim-comparison/error-percentages.csv) · [Seeds, paired differences and intervals](results/screen-run-2026112102/stim-comparison/comparison.json)

From a repo environment installed with `python -m pip install -e .`, rerun this comparison against the saved checkpoint with the following command. Supply a fresh output directory. New runs of `main.py` also save the percent chart under `stim-comparison/`.

```sh
python new-approach/compare_stim_fft.py --run new-approach/results/screen-run-2026112102 --out new-approach/results/my-stim-comparison --logical-shots 100000 --seed 2026112103
```

## Original raw-check CNN comparison

The experiment asks whether short readout glitches and persistent data faults can be separated in frequency space well enough to locate the faulty qubits. On **10,000 held-out Stim shots**, filtering removed **57.4% of readout-residual power** and retained **63.1% of clean data-check power**. The filtered CNN made **35 fewer site mistakes** than the raw-check CNN across 250,000 decisions. The paired confidence interval includes zero, so a repeatable localization advantage remains unproven.

## Follow one shot

These figures follow held-out shot 1, counting from zero. It was the first shot on which the filtered CNN flagged a site. Selection used the CNN calls before consulting the injected faults. Aggregate results below include every test shot.

| View | What to look for |
| --- | --- |
| [Five rounds of Stim measurements and their spatial spectra](results/fft-five-rounds/frequency_by_time_step.png) | Follow the same patch from rounds 1 through 5. Each lower panel shows the spatial FFT of the measured checks above it. |
| [Weighted frequency stack before and after filtering](results/fft-five-rounds/frequency_stack_filter_comparison.png) | Later rounds contribute more weight to the estimate of the final state. Both panels share an amplitude scale. |
| [Wavenumber versus temporal frequency](results/fft-five-rounds/wavenumber_frequency.png) | The two axes distinguish changes across the patch from changes between rounds. Each panel uses the same power scale. |
| [Actual and selected qubits](results/fft-five-rounds/qubit_selection_example.png) | The CNN flagged Stim IDs **1, 3 and 18**. ID **1** was correct, IDs **3 and 18** were false positives, and ID **7** was missed. |
| [Error differences on all held-out shots](results/fft-five-rounds/cnn_site_error_comparison.png) | Compare false calls and misses per 100 shots. Each method saw exactly the same measurements. |

The example also has a [machine-readable list of Stim qubit IDs, grid positions, circuit coordinates and CNN scores](results/fft-five-rounds/example-qubits.json).

## What Stim generates

The default circuit is a rotated distance-5 Z memory with 25 data qubits and 12 Z checks. Each round independently applies an X fault to every data qubit with probability **0.008**, and flips each ancilla readout with probability **0.04**. The remaining circuit operations are ideal. Five rounds yield five grids of raw Z-check outcomes. A data X fault can change neighboring checks in several consecutive rounds. A readout flip changes one recorded outcome. A real fault first appearing in round five can also look brief.

A target site has an **odd number of injected X faults** by the end of round five. Two X faults on one qubit cancel. A shadow Stim run with identical data faults and ideal readout provides clean checks for measuring filter performance. Inference takes only observed checks and the known lattice geometry. Final data measurements and injected fault records stay outside the CNN inputs.

## What the FFT and CNN do

1. Place each round's measured checks on a 6 × 6 grid. The 12 physical Z-check positions are marked in a geometry mask. Compute the spatial FFT on a zero-padded 12 × 12 grid. The added bins sample the spectrum more densely.
2. Reflect the five observed frames in reverse order to form a ten-frame transform window. This avoids directly joining the last observation to the first. The five observed rounds set the available time information.
3. Retain modes with temporal frequency at most **0.20 cycles per round** in magnitude and radial spatial wavenumber at most **0.48 cycles per grid cell**. Keep the zero-frequency component so a persistent signal can pass. For the stack, sum **complex** spatial coefficients with successive weights **1/15, 2/15, 3/15, 4/15, 5/15**. Averaging before taking the magnitude preserves phase.
4. Invert the filtered spectrum to put the evidence back at its physical positions. The CNN takes five reconstructed grids, their weighted sum, and the geometry mask. A shared 4 × 4 convolution with 24 ReLU channels and a pointwise sigmoid readout gives each data qubit a score. A validation threshold converts scores into selected Stim qubit IDs.

The raw-check control uses the same architecture, weights at initialization, training shots, minibatch order and epoch budget. Both models start from one checkpoint pretrained on **8,000** independently sampled shots with ideal readout. Each fine-tunes on the same **16,000** noisy shots. Each chooses its own threshold by maximizing positive-site F1 on the same **4,000** validation shots, balancing precision and recall. Only then is the final test split drawn.

## Results and interpretation

| Input to CNN | False calls | Missed sites | Site error rate | Precision | Recall | Entire faulty set correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw checks | 3,820 | 2,461 | 2.5124% | 65.48% | 74.65% | 61.78% |
| FFT-filtered checks | 3,794 | 2,452 | 2.4984% | 65.67% | 74.74% | 61.82% |

Site error rate counts false calls plus missed faults across all 25 sites per shot. The filtered model found **7,256 of 9,708** true faulty sites. Its site error rate changed by **−0.0140 percentage points**, with a paired 95% interval of **−0.0461 to +0.0181**. The interval uses the between-shot variation in paired error counts, preserving dependencies among qubits in a shot.

For the power diagnostic, the clean data-check signal is the ideal-readout record. The readout residual is the signed difference between the observed and clean records. For each component, divide pooled in-band Fourier power by pooled total Fourier power across the test split. Some real data signal lies outside the passband, and some readout noise lies inside it. The measured power reduction therefore translates to only a small observed change in site errors. This experiment measures physical X-fault localization under the stated controlled noise model. A competition logical-error rate would require decoding and comparing logical observables under the competition circuit and noise rules.

## Run and verify

From the repository root, use Python 3.11+ with NumPy, SciPy, Matplotlib and Stim installed. For this saved run the versions were Python 3.11.4, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2 and Stim 1.16.0.

```sh
python new-approach/main.py --out new-approach/results/my-five-round-run
python new-approach/verify_run.py new-approach/results/fft-five-rounds
python -m pytest -q new-approach
```

The first command requires a fresh output directory. The verifier regenerates the held-out Stim shots from the saved seed, reloads both CNNs, checks their site selections, and independently recounts the errors. Seeded replay should use the recorded dependency versions. [Experiment settings, seeds and scores](results/fft-five-rounds/metrics.json) · [Site-error CSV](results/fft-five-rounds/site-errors.csv) · [Held-out measurements, labels and CNN scores](results/fft-five-rounds/held-out.npz)

For inference from Python in this directory, pass five binary check grids per shot in Stim's row-major check layout. The returned `selected` grid has one boolean per data qubit.

```python
from main import predict_sites

scores, selected, stim_ids = predict_sites("results/fft-five-rounds", observed)
first_shot_qubits = stim_ids[selected[0]]
```
