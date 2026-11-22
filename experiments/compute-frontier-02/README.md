# Training decoders for accuracy and throughput

This study tests whether offline training can identify the shots that benefit from correlated matching. A second experiment tests whether belief matching can resolve disputes between ordinary and correlated matching. The circuits use the team's provisional Stim noise model. Each experiment stores one logical Z bit through five or seven check rounds at physical error parameter 0.005. The final test contains 300,000 complete memory records per distance.

[Held-out results and timing figure](summary.md) · [Results per decoder](accuracy.csv) · [Paired comparisons](paired.csv) · [Timing samples](timing.csv) · [Worker scaling](scaling.csv)

At distance 7, the learned policy scored **0.9067%** logical error at **9.069 µs/shot** in serial batches. Ordinary matching scored **0.9973%** at **11.667 µs/shot**. With eight workers on the same host and 8,192-shot batches, they took **1.483** and **1.909 µs/shot**, respectively. Comparing learned routing to the previous activity-only rule at this distance, the paired accuracy test gave p = 0.0172, above the prespecified 0.0125 cutoff. Its batch speed relative to that rule was also inconclusive. At distance 5, learned routing passed the adjusted accuracy test against the earlier rule.

The lowest observed distance-7 error was **0.5310%** with disagreement rescue. It evaluated the belief referee on **0.808%** of shots. Full correlated matching scored **0.5890%**, and full belief matching scored **0.5503%**. Distance 5 favored full belief matching for absolute error, at **0.9757%** versus **1.0053%** for disagreement rescue. These comparisons use the same shots within each distance. The checked-in records can be independently regenerated and audited with the command below.

## Follow-up comparisons

For an individual live record, the dedicated bytes-to-bit learned decoder reduced median distance-7 latency to **14.20 µs**. A direct ordinary matcher took **17.10 µs** on the same records. The 95th percentiles were **71.00** and **36.35 µs**, respectively. The learned method is most attractive for batch throughput and median latency. [Single-shot study and reproducer](../compute-scalar-01/README.md)

Two independent 180,000-shot distance-7 studies screened cheaper belief referees. A four-iteration product-sum referee took **50.51 µs/shot** and scored **0.5550%**, compared with **76.56 µs/shot** and **0.5394%** for the twelve-iteration reference on the same records. An eight-iteration minimum-sum referee also saved time, while scoring more failures than the twelve-iteration reference on its separate holdout. Both cheaper choices made fewer errors than always-correlated matching on their respective holdouts. [Product-sum budget comparison](../specialist-budget-01/results.md) · [Minimum-sum budget comparison](../specialist-budget-accuracy-01/results.md)

## How the policies work

**Learned routing.** Generate 1.2 million training records per circuit. Score ordinary matching on the graph components that carry the memory observable, and correlated matching on the whole record. For each training record, count whether correlated matching fixes an ordinary error, introduces an error, or leaves correctness unchanged. Fit separate regularized models for fixes and regressions. Their difference estimates the benefit of escalation. The trained policy compiles these estimates into lookup tables. Runtime features summarize detector activity, the occupied spatial strips along the logical direction and their overlap in adjacent rounds. At distance 5 the model also uses a parity feature across those strips. The rule targets roughly 5.2% escalation. Labels are only used offline. The circuit and trained policy have stored SHA-256 checksums.

**Fast ordinary branch.** Most records in the learned route go to ordinary matching on the observable-bearing graph components. When the model is sparse enough, a bounded table stores that matcher's answers for common projected syndromes. A lookup accepts a result only after comparing the full packed record. Misses go to the original matcher. The heuristic decides when the cache is worth constructing. It preserves the matcher's answers.

**Disagreement rescue.** Compute ordinary matching on the observable-bearing components and correlated matching on the full syndrome. When they disagree, ask belief matching to make the final prediction. It can run up to 12 iterations of sum-product belief propagation on the fault model and then use its matching fallback. The method can reduce errors at the cost of additional inference. The learned-routing-plus-rescue variant only requests a correlated vote for shots selected by the learned rule.

**Logical-gap control.** A boundary parity constraint tests whether the competing logical answer is within two units of ordinary graph cost. An auxiliary edge caps the work of searching for a costlier alternative. Records judged close go to correlated matching. The gap is an established reliability measure. This study tests a particular capped-cost implementation in these circuits. [Research on soft-output decoding](https://arxiv.org/abs/2405.07433)

**Controls.** Ordinary MWPM, always-correlated MWPM, the earlier activity-count rule and full belief matching score the same held-out syndromes. An additional random control sends exactly the same number of records per seed to correlated matching as the learned policy. The random control has accuracy results only. It measures the value of choosing the records.

## What was measured

The protocol fixes the methods, feature choices, routing budgets, seeds and four primary accuracy tests before scoring. Training, calibration, scoring and timing use separate seed ranges. The archived decisions come from combining the same detector-only branch outputs for all candidates. The runner checks the combined decisions against the individual deployed decoders on the first 256 records of every final seed. The independent replay tool can also verify the deployed learned decoder across all records.

Serial batch timing uses eleven independently sampled batches of 2,048 records in interleaved random decoder order. Single-record times are sampled independently. Worker scaling uses nine 8,192-record batches at distance 7. All four scaling methods use the same records and the same worker configurations. Each worker owns an initialized decoder. Timed parallel calls include sending input, decoding and collecting results in shot order. Pool startup, warm-up, Stim sampling and offline training are outside the inference timer. The scaling study checks every returned prediction against serial execution. Measurements were made on a Windows host with an Intel Core i7-13700HX. An organizer-supplied circuit requires its own training and test.

## Run the saved decoder

Run from the repository root with the project environment active. The installation uses the project's existing development and hypergraph extras. The input file is a sequence of complete little-endian Stim b8 detector records for the named circuit. The output is one byte per predicted logical bit. Choose a fresh output path.

```sh
python -m pip install -e ".[dev,hypergraph]"
python -m bascd.compute_replay predict --case experiments/compute-frontier-02/d7 --method learned --input detectors.b8 --output predictions.b8
python -m bascd.compute_replay audit --run experiments/compute-frontier-02 --verify-learned
```

To reproduce the study with the stored recipe, start with a fresh directory. Use the recorded dependency versions for seeded replay.

```sh
python -m bascd.compute_campaign prepare --out results/my-compute-study
python -m bascd.compute_campaign score --out results/my-compute-study
python -m bascd.compute_campaign time --out results/my-compute-study
python -m bascd.compute_campaign scale --out results/my-compute-study
python -m bascd.compute_report --out results/my-compute-study
```

[Frozen protocol](protocol.json) · [Distance-5 policy hash and training time](d5/freeze.json) · [Distance-7 policy hash and training time](d7/freeze.json)
