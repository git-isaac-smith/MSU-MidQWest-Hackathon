# Basis-specific bath prior on fixed Stim records

The frozen correlated-matching rule uses the fixed-clock bath prior for X memory and the expanded equal-axis prior for Z memory. The primary reference always uses equal-axis weights. The secondary reference always uses bath weights. The basis comes from the known circuit configuration. Both graphs use the same expansion procedure for round-start data idles.

The comparison uses the original provisional `simple_uniform` circuits across distances 3, 5 and 7, both memory bases, and physical error probabilities 0.001, 0.003, 0.005 and 0.01. Rounds equal distance. Root seed 562491030 generates 125,000 fresh shots per scenario for 3 million paired shots. The manifest records the fixed rule and seeds before sampling. Each matcher receives the same detector bits. Only the evaluator reads Stim's sampled logical observable.

The direction of the rule follows the [previous confirmation](../bath-confirmation-01/README.md), where the bath prior recovered 579 net failures in X memory and added 189 in Z memory against equal-axis matching.

## Held-out results

| Correlated-matching policy | Logical failures / 3,000,000 | Logical error | Warm batch µs/shot |
| --- | ---: | ---: | ---: |
| Bath in X, equal-axis in Z | 67,772 | 2.25907% | 17.04 |
| Always equal-axis | 68,407 | 2.28023% | 17.83 |
| Always bath | 67,795 | 2.25983% | 16.99 |

Compared with always using equal-axis weights, the frozen rule recovered 4,595 failures and introduced 3,960, for a reduction of 635 failures. The paired difference was −0.02117 percentage points with a scenario-stratified approximate 95% interval of [−0.02721, −0.01512]. That change comes from X memory, where the bath prior reduced error by 0.04233 points on 1.5 million shots.

Compared with always using bath weights, the rule recovered 3,262 failures and introduced 3,239, for a reduction of 23 failures. The difference was −0.00077 points with a 95% interval of [−0.00603, +0.00450]. The comparison leaves the benefit of switching to equal-axis weights in Z memory unresolved.

Timings are descriptive. Each graph was warmed and timed on one full batch per scenario. Equal-axis ran first in X memory and bath ran first in Z memory, so the selected graph ran second throughout. The policy timing uses the selected graph's measured batch time, reflecting that the basis is known at setup.

## How low can the fixed workload go?

For each distance-3 memory, the project's posterior decoder can enumerate all 2²⁴ possible detector records. Some records occur with both possible logical outcomes. Even a decoder that always picks the more likely outcome has a positive expected error rate. For distance-3 Z memory at `p=0.005`, the calculated optimum is 1.44427%. [Posterior calculation](../../src/bascd/exact_small.py) · [Circuit and stored estimate](../circuit-frontier-01/portable-d3/manifest.json)

On the equally weighted 24-scenario workload, summing those eight distance-3 optimum rates and dividing by 24 gives **0.61716%**. This is a numerical estimate of a lower bound on expected pooled error for a decoder using only these detector records, even allowing perfect decisions at distances 5 and 7. A finite sample can fluctuate below its expectation. To check the sum, select the eight `distance == 3` scenarios and use `diagnostics.exact_small.estimated_optimal_ler` for each. The bath timing experiment has its own physical channels and measured erasure locations. [Eight per-circuit estimates](../iteration-04/accuracy/validation.json) · [Fresh replay of Sakidja's timed checks](../sakidja-table5-replay-01/README.md)

## Reproduce and verify

Run from the repository root with the project installed in the active Python environment. Supply a fresh output directory for a new run.

```powershell
python experiments/bath-basis-policy-01/run.py experiments/bath-basis-policy-01/reproduction
python experiments/bath-basis-policy-01/run.py experiments/bath-basis-policy-01/confirmation --replay
```

The verifier checks the paired counts and pooled and basis summaries and regenerates the circuit and sample fingerprints from the saved seed plan. See the [manifest](confirmation/manifest.json), [scenario CSV](confirmation/scenarios.csv), and [summary](confirmation/summary.json).
