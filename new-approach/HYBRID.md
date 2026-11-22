# A logical decoder that asks for a second opinion

The decoder starts with ordinary PyMatching. It estimates how much more likely one logical answer is than the other, and spends additional computation on ambiguous shots. Its input is a Stim detector record. A logical error means the decoder predicted the wrong final logical bit. All reported error percentages use shots as the denominator.

## Measured results

**The balanced hybrid reduced the logical error rate from 0.4370% to 0.4314% on 1,500,000 held-out shots.** It fixed 291 matching mistakes and introduced 207, for 84 fewer failures. The reduction was **0.0056 percentage points**, with a paired 95% interval of **0.0027 to 0.0085 percentage points**. This is a 1.28% relative reduction. The hybrid made fewer errors in each of the three test splits.

| Detector-only policy | Wrong logical bits | Logical error | Sent for exact inference | Batch µs/shot |
| --- | ---: | ---: | ---: | ---: |
| Ordinary PyMatching | 6,555 | 0.4370% | 0% | 1.93 |
| Matching + raw-input CNN | 6,537 | 0.4358% | 0% | — |
| Matching + FFT CNN | 6,545 | 0.4363% | 0% | 23.92 |
| Matching margin + selective exact inference | 6,475 | 0.4317% | 0.5251% | 25.29 |
| **Balanced FFT hybrid** | **6,471** | **0.4314%** | **0.5255%** | **30.54** |
| Matching + exact inference throughout the shortlist | **6,461** | **0.4307%** | 5.4236% | 66.29 |

The CNN-free matching-margin control fixed nearly as many errors at the same routing budget. The balanced hybrid's four-failure advantage over that control has a paired 95% interval spanning zero. The FFT-input network also had eight more failures than the raw-input network when both used only learned overrides. Selective probability calculations account for most of the demonstrated gain. The lowest-error tested policy was the last row, which skips the CNN and spends more time resolving ambiguous shots. These results apply to the controlled noise settings below.

[Percent error chart](results/hybrid-01/hybrid-errors.png) · [Machine-readable results and paired intervals](results/hybrid-01/results.json) · [Counts and timings](results/hybrid-01/results.csv) · [Locked policy](results/hybrid-01/policy.json) · [Epoch history](results/hybrid-01/training.json)

## How a decision is made

1. **Match the detectors.** Compute the normal MWPM logical prediction. Then constrain the spatial boundaries to calculate the cheapest correction compatible with each of the two possible logical bits. Their cost difference is the matching margin. The two costs describe the best explanation for each answer, while several plausible explanations can support the same answer.
2. **Reconsider close calls.** When the absolute margin is at most 5.3 log-odds units, a logical CNN estimates the probability that the matching answer is wrong. It sees the five raw check rounds, phase-preserving filtered Fourier fields, predictions of the earlier FFT site CNN, public final check parities and matching features. The network starts from the log odds implied by the matching margin and learns a correction. It can reverse the predicted logical bit.
3. **Resolve uncertainty.** The balanced policy sends the CNN's most uncertain cases to an exact probability calculation. That calculation adds the probabilities of compatible fault histories for each logical bit and returns the more probable bit. The cutoff was set on validation data to route about 0.5% of all shots. Confident shots keep the matching or CNN decision. Exact numerical ties retain the ordinary matching answer.

The exact calculation uses the 12 Z-check parities and one logical parity for a distance-5 rotated memory, giving 8,192 possible states per round. It propagates their probabilities under the known data-fault and readout probabilities. Surface-code distances beyond 5 and richer circuit-noise channels would require a different implementation or approximation.

## Training and controls

The experiment uses distance 5, five rounds, independent data X faults with probability 0.008 per qubit per round, independent ancilla readout flips with probability 0.04, and ideal final readout. The final *check parities* are public. Stim samples the logical answer into a separate array used for evaluation.

The training sampler drew **2,400,000 shots**. It kept every case with margin at most 3 and a seeded 25% sample of other cases up to margin 5.3. Each of the **58,824 retained examples** received a training target equal to the exactly calculated probability that matching was wrong. This gives the network more information than a single sampled right-or-wrong label for the same syndrome. A separate 300,000-shot validation set determined the checkpoints, override thresholds and routing cutoffs. Both the Fourier network and its raw-input control trained for 40 epochs with identical seeds and budgets. Checkpoint selection used validation cross-entropy. Override threshold selection minimized expected logical error on validation.

| Evaluation policy | What it measures |
| --- | --- |
| PyMatching | The standard Stim tutorial's MWPM baseline. |
| Raw checks + CNN | Matching plus a logical CNN trained with the Fourier inputs and pretrained FFT site evidence removed. |
| FFT features + CNN | Matching plus the trained Fourier CNN, with no exact calculations at inference. |
| Matching margin + exact | A control that uses matching margin alone to route about 0.5% of shots for exact evaluation. |
| Balanced FFT hybrid | The primary candidate. It uses the learned logical decision and selectively computes an exact answer. |
| Exact on ambiguous shots | An accuracy-oriented control. It uses the exact answer throughout the shortlisted matching-margin range and ordinary matching elsewhere. |

Each pair receives identical detector records. Three fresh 500,000-shot seeds measure performance after the policy is locked. The report gives paired gains and losses, a 95% interval for the change in logical error, and separate inference timings. Timings use a fourth seed, warm up the decoder, measure batches of up to 4,096 shots, and take the median of three passes. They include public-feature extraction and any exact calculations required by the measured policy. They exclude training, circuit sampling, and decoder construction.

## Reproduce and run

From the repository root with the project installed (`python -m pip install -e .`), use a fresh directory to repeat training and evaluation.

```sh
python new-approach/hybrid_study.py --out new-approach/results/my-hybrid-run
python new-approach/hybrid_study.py --stage verify --out new-approach/results/hybrid-01
```

For deployment on the circuit's Stim `b8` detector file, the output is one predicted observable bit per shot. The policy directory includes both trained CNNs, the earlier site checkpoint and the exact circuit definition.

```sh
python new-approach/hybrid_decode.py --policy new-approach/results/hybrid-01 --detectors input.b8 --out predictions.b8 --mode balanced
```

The same decoder is available in Python from `new-approach`.

```python
from hybrid_policy import HybridDecoder

decoder = HybridDecoder.load("results/hybrid-01")
predicted_logical_flips = decoder.predict(public_detector_bits, mode="balanced")
```
