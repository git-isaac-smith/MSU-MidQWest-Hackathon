# Scaling the five-round FFT and posterior prototypes

We ran the archived distance-5 site CNN checkpoints on fresh Stim records from distance 5 through 33, growing from 25 to 1,089 data qubits. The circuits store one logical Z bit for five noisy check rounds. Each round applies independent data X faults at 0.8% and check readout flips at 4%. The final data readout is ideal. This controlled model is the circuit on which the FFT prototype was trained and is separate from the circuit-level benchmark.

Each method at a given distance saw the same public detector records. Stim returned the true logical flips separately. The raw-check and FFT site CNNs used their saved distance-5 weights and thresholds at every distance. Shared convolutions let the checkpoints accept larger check grids directly. We tested two outputs from each site CNN. Site parity uses the five noisy check grids. Site parity with terminal matching also uses the ideal terminal check parities to repair the residual syndrome with the prototype's existing spatial matcher. Ordinary MWPM and list matching use the complete detector record.

| Method | d5 · 25 qubits | d9 · 81 qubits | d17 · 289 qubits | d33 · 1,089 qubits |
| --- | ---: | ---: | ---: | ---: |
| Ordinary MWPM | 13/4,096 | 1/1,024 | 0/1,024 | 0/1,024 |
| Raw site CNN, parity | 594/4,096 | 252/1,024 | 378/1,024 | 463/1,024 |
| FFT site CNN, parity | 611/4,096 | 264/1,024 | 371/1,024 | 462/1,024 |
| Raw site CNN with terminal matching | 69/4,096 | 6/1,024 | 0/1,024 | 0/1,024 |
| FFT site CNN with terminal matching | 74/4,096 | 5/1,024 | 1/1,024 | 0/1,024 |
| Matching with alternative corrections | 12/4,096 | 1/1,024 | 0/1,024 | 0/1,024 |
| Raw or FFT logical CNN override, each | 13/4,096 | — | — | — |
| Balanced FFT logical hybrid | 12/4,096 | — | — | — |
| Selective posterior, margin-only or full shortlist, each | 12/4,096 | — | — | — |
| Exact posterior on every shot | 12/4,096 | — | — | — |

The FFT site CNN's direct logical parity error rose from 14.92% at d5 to 45.12% at d33. Adding terminal matching reduced the observed error to 1.81% at d5 and zero failures at d33. Each zero-failure result on 1,024 shots has a Wilson 95% upper bound of 0.374%. These samples are too small to distinguish low error rates at the largest distances. The logical CNN uses a fixed d5 readout over spatial positions. The existing exact posterior implementation is also restricted to small systems.

Matching with alternative corrections used the frozen d7 recipe at each distance. A matching-cost gap of at most 5.3 triggers a search over candidates from 49 matchers. The trigger fired on 209, 3, 0 and 0 records at d5, d9, d17 and d33, respectively. The large-distance runs exercised the gap probe, with every decision there following the ordinary matching branch.

## Inference timing

The comparison page reports median microseconds per shot beside each error result. We timed five distinct 64-record chunks after a warm-up, using interleaved decoder order on an Intel Core i7-13700HX with one process and BLAS/OpenMP limits of one. Timing begins with unpacked detector arrays and excludes setup, sampling, unpacking and scoring. All timed predictions matched the accuracy archive. The raw logical CNN timing includes the prototype's feature construction and feature ablation. Machine load, batch size and routing frequency can change the measured values. [Accuracy and runtime table](index.html) · [Individual batch times](measured/timing.csv)

## Archived accuracy and runtime

The earlier studies used larger holdouts under the same controlled model. Their runtime measurements come from their own benchmark runs.

| Archived comparison | Paired holdout | Logical error | Warm batch µs/shot |
| --- | ---: | ---: | ---: |
| d5 balanced FFT hybrid vs MWPM | 1,500,000 | 0.4314% vs 0.4370% | 30.54 vs 1.93 |
| d5 compiled full posterior vs MWPM | 1,500,000 | 0.42627% vs 0.43167% | 11.00 vs 1.92 |
| d5 archived posterior cache and selective exact inference vs MWPM | 1,500,000 | 0.43067% vs 0.43220% | 2.14 vs 1.86 |
| d7 matching with alternative corrections vs MWPM | 1,500,000 | 0.11447% vs 0.11973% | 77.56 vs 3.94 |
| d7 radius-two approximate posterior, taking all proposed overrides, vs MWPM | 450,000 | 0.21067% vs 0.11844% | — |

The d7 radius-two study chose plain matching on development data. The above approximate-posterior row reflects its held-out aggressive override control. The cached-posterior row converts the original median time per 49,152-shot batch to microseconds per shot. [FFT hybrid](../../../new-approach/HYBRID.md) · [Full posterior and d7 comparisons](../../../new-approach/POSTERIOR.md) · [Archived posterior cache](../../../new-approach/results/cached-teacher-01/README.md)

## Reproduce the comparison

From the repository root, use the project environment and the existing native posterior library in `build/`. The build instructions are in [native_posterior.py](../../../new-approach/native_posterior.py).

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.venv/Scripts/python.exe experiments/qubit-scaling-01/fft/run.py --out experiments/qubit-scaling-01/fft/replay
.venv/Scripts/python.exe experiments/qubit-scaling-01/fft/runtime.py --run experiments/qubit-scaling-01/fft/replay --out experiments/qubit-scaling-01/fft/replay/timing.csv
```

The run records its seeds, versions and checkpoint hashes. It saves each Stim circuit and the exact public detector rows, truth, and predictions used for scoring. The CSV includes shot counts, Wilson intervals, and paired recoveries and regressions against MWPM. [Run manifest](measured/run.json) · [Results CSV](measured/accuracy.csv)

