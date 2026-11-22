# How much logical error can we remove?

Each shot asks one question. Given only the public detector record, what is the final logical bit? Stim supplies the answer separately for scoring. The error percentage is 100 times the number of wrong answers divided by the number of shots. All comparisons within a distance use the same shots.

These experiments use the team's controlled Z-memory circuit with five rounds, independent X faults on data qubits at 0.8% per qubit per round, independent check readout flips at 4%, and ideal final data readout. The organizer's supplied brief leaves the competition noise prescription unspecified. The percentages below apply to these simulations.

**The lowest observed error at distance 7 was 0.11447%.** Matching with alternative corrections made 79 fewer logical mistakes than ordinary PyMatching on the same 1.5 million shots. At distance 5 the full posterior calculation reproduced the previous accuracy mode's decisions 6.1 times faster. [Comparison graph](results/list-matching-01/decoder-comparison.png)

## Distance 5

We kept the same 25 data qubits, noise, rounds, and previously trained FFT model, and sampled three fresh seeds with 500,000 shots each. The new decoder sums the probabilities of all compatible fault histories for each final logical bit. It returns the more probable bit, using matching to break numerical ties. It does this for every shot. The prior accuracy mode only calculated that sum when matching found two competing answers close in cost.

| Decoder | Wrong / 1,500,000 | Logical error | Warm batch µs/shot |
| --- | ---: | ---: | ---: |
| Ordinary PyMatching | 6,475 | 0.43167% | 1.92 |
| Previous balanced FFT-CNN hybrid | 6,383 | 0.42553% | 32.06 |
| Previous posterior on ambiguous shots | 6,394 | 0.42627% | 67.11 |
| New posterior on every shot, compiled | 6,394 | 0.42627% | 11.00 |

The new decoder corrected 302 matching mistakes and introduced 221, a net reduction of 81. The paired 95% interval for its change against matching was −0.00839 to −0.00241 percentage points. The two posterior modes agreed on all 1.5 million records. The balanced hybrid made 11 fewer mistakes on this sample. Only 39 predictions differed between it and the new decoder, and the exact paired test gave *p* = 0.108.

For known noise probabilities, choosing the more probable logical bit minimizes the expected number of wrong answers given the available checks. The exact calculation therefore provides a model-specific reference for future CNNs. A finite sample can favor another decoder on a few ambiguous shots. Here the implementation improvement was speed. Caching shared histories and combining forward and backward probabilities over 8,192 states reduced batch time by a factor of 6.1 compared with the previous accuracy mode. The timing uses three batches of fresh records after warm-up and excludes sampling and setup. [Paired counts, intervals and rates](results/full-map-01/results.json) · [Timing protocol](results/full-map-01/implementation.json) · [Distance-5 graph](results/full-map-01/full-map-errors.png) · [Background on maximum-likelihood surface-code decoding](https://arxiv.org/abs/1405.4883)

## Reproduce the distance-5 results

Run these commands from the repository root in the project Python environment. The example uses the optional compiled backend used for the timings above. Build it for your CPU using the platform command at the top of [native_posterior.py](native_posterior.py). The portable NumPy backend is available with `--backend cached` for the study or `--posterior-backend cached` for deployment.

```sh
python new-approach/full_map_study.py --out new-approach/results/my-full-map --backend native
python new-approach/full_map_study.py --out new-approach/results/full-map-01 --backend native --stage verify
python new-approach/hybrid_decode.py --policy new-approach/results/hybrid-01 --mode full_map --posterior-backend native --detectors detectors.b8 --out predictions.b8
```

The last command reads packed public detectors in the saved circuit's order and writes one predicted observable per shot. A separate scorer holds the true bits. The study saves the circuit, seed plan, public records, true bits, predictions and posterior probabilities. Verification regenerates all records and checks the original NumPy posterior independently on a subset. [Stim circuit](results/full-map-01/benchmark.stim) · [Seeds and versions](results/full-map-01/plan.json) · [Results CSV](results/full-map-01/results.csv)

## Distance 7

Increasing distance from 5 to 7 uses 49 data qubits in place of 25. We kept five rounds and the same physical fault probabilities. Comparing two decoders on identical distance-7 shots measures a decoder improvement. Comparing distances also changes how much redundancy protects the bit.

### Matching with alternative corrections

The baseline picks a lowest-cost explanation of the detector record. When matching finds plausible explanations for both final bit values, several different corrections can support the same answer. Our decoder runs ordinary matching first. If the cost gap between the two possible answers is at most 5.3, it searches both classes with 49 matchers. Forty-eight use small or large reproducible changes to the search weights to find alternate corrections. It counts each distinct correction once and assigns it the probability given by the original detector error model. The class with the larger summed probability wins. This is an approximation because a finite list can miss corrections. No CNN or LLM inference is needed.

The decision threshold was selected on 500,000 development shots and fixed before three fresh test samples were drawn. Both decoders saw the same public detector bits and neither saw the true logical bit. Of 1.5 million final shots, 19,894 (1.33%) required the extra search. A second run regenerated all two million development and test records from their seeds, reproduced the predictions and recounted the errors. The deployment command also reproduced the saved decisions on 10,000 held-out records in Stim's packed detector format.

| Test seed | Shots | Ordinary matching errors | Alternative corrections errors |
| --- | ---: | ---: | ---: |
| 84374101 | 500,000 | 570 | 563 |
| 84374102 | 500,000 | 613 | 572 |
| 84374103 | 500,000 | 613 | 582 |
| **Total** | **1,500,000** | **1,796 (0.11973%)** | **1,717 (0.11447%)** |

The new decoder fixed 190 errors and introduced 111. That reduced the error rate by **0.00527 percentage points**, a **4.40% relative reduction**. The paired 95% interval for the reduction was **0.00300 to 0.00753 percentage points**. The exact paired test gave *p* = 0.00000617. The new decoder's own Wilson 95% interval was 0.10918% to 0.12001%. The paired test captures that both methods almost always agree on individual shots. [Per-seed counts and paired statistics](results/list-matching-01/results.csv) · [Test protocol](results/list-matching-01/protocol.json) · [Development scores and chosen rule](results/list-matching-01/policy.json)

The three decoding stages together averaged **77.56 µs per shot in batches**, including ordinary matching, checking the cost gap and resolving ambiguous cases. Ordinary matching alone averaged **3.94 µs/shot**. The stand-alone Python calls on 4,096 separate fresh records had a median latency of **105.7 µs** for the new decoder and **10.3 µs** for matching. At the 99th percentile they took **3.52 ms** and **29.1 µs**, respectively. These host measurements exclude generating Stim samples. [Stage timings](results/list-matching-01/results.json) · [Single-shot latency measurements](results/list-matching-01/latency.json)

From the repository root, the following commands reproduce the study in a fresh directory, replay the saved study, or apply the saved policy to a detector-only Stim file.

```sh
python new-approach/distance7_study.py run --candidate list-matching --out new-approach/results/my-list-matching
python new-approach/distance7_study.py verify --run new-approach/results/list-matching-01 --replay
python new-approach/hybrid_decode.py --mode distance7 --policy new-approach/results/list-matching-01 --detectors detectors.b8 --out predictions.b8
python new-approach/measure_posterior_latency.py --backend native --out my-latency.json
```

### Testing a limit on readout faults

This candidate calculates the probabilities of all possible data-fault transitions over one round. To make five-round inference tractable, it considers up to two readout flips in each of the first four rounds. The fifth readout adds no information once the ideal terminal check parities are known under this specific noise model. Matching routes shots with a cost difference of at most 5.3 to the candidate.

On 150,000 development shots, ordinary matching made 190 mistakes. Using every proposed override made 305. All five override thresholds tested increased the count. The development rule therefore selected plain matching before the final samples were drawn. On 450,000 new shots ordinary matching scored **533 wrong (0.11844%)**, with a Wilson 95% interval of 0.10881% to 0.12893%. Taking every override would have raised this to **948 wrong (0.21067%)**. The candidate fixed 109 matching errors and introduced 524. This failure is evidence against the readout cutoff as a reliable approximation for this circuit. [Development policy](results/distance7-01/policy.json) · [Held-out results](results/distance7-01/results.csv)

```sh
python new-approach/distance7_study.py run --radius 2 --out new-approach/results/my-distance7
python new-approach/distance7_study.py verify --run new-approach/results/distance7-01 --replay
```
