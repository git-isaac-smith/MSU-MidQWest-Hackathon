# Million-shot bath stress sweep

The fitted balanced check had a lower logical error rate at all five levels. At the native 0.1% background it had 152 failures in one million shots, a 0.01520% rate. The earlier independent run had 74 failures in 500,000 shots, a 0.01480% rate. [Million-shot counts and intervals](result.json) · [Earlier confirmation](../bath-rare-confirmation-02/result.json)

| Added X and Z error probability, each | Fixed clock failures | Fixed rate and 95% Wilson interval | Fitted check failures | Fitted rate and 95% Wilson interval |
| ---: | ---: | ---: | ---: | ---: |
| 0.1% | 170,139 | 17.01390% [16.94038%, 17.08767%] | 152 | 0.01520% [0.01297%, 0.01782%] |
| 0.5% | 222,318 | 22.23180% [22.15041%, 22.31340%] | 1,796 | 0.17960% [0.17149%, 0.18809%] |
| 1% | 277,864 | 27.78640% [27.69869%, 27.87428%] | 7,097 | 0.70970% [0.69344%, 0.72634%] |
| 2% | 366,315 | 36.63150% [36.53712%, 36.72598%] | 34,062 | 3.40620% [3.37083%, 3.44193%] |
| 4% | 461,235 | 46.12350% [46.02581%, 46.22122%] | 157,693 | 15.76930% [15.69800%, 15.84086%] |

## Protocol

The sweep uses the ten original distance-7 rotated Z-memory stress cells. Each cell draws 1,000,000 fresh Stim shots, with 200,000 shots for each of the five original bath-to-data-qubit layouts. The five background levels add independent X and Z errors at the same probability per axis and are applied to both the sampled circuits and the decoder-prior circuits. Readout and reset stay at 0.1%. [Original stress setup](../bath-background-stress-01/README.md) · [Frozen circuits and seed schedule](plan.json)

The frame-calibrated clock uses its published bath channel as its decoder prior, matching the original oracle-calibrated comparison. The fitted balanced check uses a prior reconstructed from the fitted bath parameters. Both arms sample from the published physical channels. The decoder receives the detector records only. The scorer compares predictions to the separately sampled logical observables. [Pinned upstream circuit and decoder](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py#L389-L510)

For each layout the runner replays the original 4,000-shot stress seed through the pinned upstream decoder, checks the saved counts, and compares the bitwise predictions to the accelerated decoder. It also checks the first 128 fresh predictions per layout against upstream. The fresh totals start with the new seed schedule. The accelerated matcher uses PyMatching's sparse blossom algorithm with the upstream graph, herald conditioning, and weight quantization. [Matcher implementation and validation](../million-engine-03/)

The 95% Wilson intervals describe Monte Carlo precision conditional on the five fixed layouts. The per-chunk decode and sample wall times reflect the parallel accuracy run and concurrent machine load. The original serial timings have their own protocol and artifact. [Original serial timing](../bath-background-stress-01/timing/README.md)

## Reproduce

Run from the repository root in PowerShell. Use the package versions recorded in `plan.json`, a clean checkout of the pinned upstream repository, and the built native matcher from `million-engine-03`.

```powershell
$upstream = Join-Path $env:TEMP 'sakidja-rql-quantum-bath'
git clone https://github.com/sakidja/RQL_Quantum_bath.git $upstream
git -C $upstream checkout 94572516042297d6b6018fe76eb2f2b547e3c46f
python experiments/million-bath-03/run_safe.py $upstream --output bath-million-replay --workers 4
```

The same command resumes from the last complete 10,000-shot chunk per layout. The launcher retries atomic checkpoint writes if another Windows process briefly holds the file open. Changing the seed schedule, the fingerprinted runner, engine source, or other fingerprinted inputs requires a fresh output directory. A run without layout filters writes the pooled result when all 50 layouts are complete. The checkpoint files record the seed, failure count, outcome-conditioned detector counts, and array digests for every chunk.

Two layouts resumed after transient Windows write failures. Each recomputed chunk matched the previously staged seed, counts, detector digest, observable digest, and prediction digest. The final audit reconciled all 50 layout files, 1,000 distinct chunk seeds, and 10,000,000 fresh shots with the pooled result.
