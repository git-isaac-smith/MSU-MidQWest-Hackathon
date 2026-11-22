# Cached posterior decisions for the archived five-round memory

The fixed candidate made 23 fewer logical errors than ordinary PyMatching on 1.5 million fresh paired shots. The paired interval included zero, and the candidate had higher measured latency. These shots give no evidence of simultaneous improvement in error rate and speed for this candidate.

| Decoder | Logical errors | Logical error rate | Median time for 49,152 shots |
|---|---:|---:|---:|
| Ordinary MWPM | 6,483 | 0.43220% | 91.46 ms |
| Cached posterior and selective exact inference | 6,460 | 0.43067% | 105.01 ms |

There were 133 recoveries and 110 losses. The candidate-minus-MWPM error-rate difference was −0.00153 percentage points, with a stratified paired 95% interval of −0.00357 to +0.00050 percentage points. The exact two-sided McNemar p-value was 0.158. The measured speed ratio was 0.871× with a paired bootstrap 95% interval of 0.768× to 0.934×. The candidate hit the lookup table on 64.91% of final accuracy shots and routed 2.47% to exact inference.

The circuit is the archived distance-5 rotated Z memory with five noisy check rounds, 0.8% independent data X faults per round, 4% independent check readout flips and ideal final data readout. The lookup table stores verified 60-bit public history keys with archived full-posterior decisions. Conditioned on the terminal syndrome, the fifth noisy check likelihood cancels, and reversing the four remaining readings while translating by the terminal preserves the posterior. Cache misses go through ordinary MWPM, which returns a prediction and its matching cost. A fixed rule then sends cache misses with cost at least 26.47 to the existing native posterior, retaining MWPM on numerical ties. The cache excludes teacher ties and uses no archived true logical labels.

The cache size and reversal rule were compared on a separate pilot. An archive fold, evaluated using public histories and teacher probabilities, estimated the expected benefit of the route independently of its runtime pilot. The frozen plan records the three final seeds, timing seed, code hash, teacher archive hashes and native library hash. The pilot reused the same short batch across passes. This allowed the stateful posterior cache to reuse more work than the final timing did. Final timing used nine distinct packed batches with alternating decoder order, a separate warm-up and equal 4,096-shot calls. Its timings include lookup, ordinary matching and exact inference. They exclude sampling, archive loading and setup. Archive loading took 0.234 s and setup took 0.441 s. The deployed teacher table occupied 16 MiB.

Run from the repository root using the archived native DLL in `build/`.

```powershell
.venv/Scripts/python.exe new-approach/cached_teacher_study.py verify
```

Verification regenerated every final detector record and truth bit, reran both decoders and reproduced all paired counts. For a new output directory, run the script phases `pilot`, `route-pilot`, `lock --table-bits 21 --no-ordered --route-cost 26.47`, then `evaluate` with the same `--out` on each invocation. [Frozen plan](plan.json) · [Paired counts, intervals and raw timing](results.json) · [Cache pilot](pilot.json) · [Routing pilot](route-pilot.json)
