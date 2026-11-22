# Holding a dual-rail excitation after its predicted return

We tested the common phase flips illustrated in Ridwan Sakidja’s *Catching the slosh* slides. First the simulated controller waits for the fitted return time and checks whether the excitation occupies the two rails. After a pass, both rails receive the same instantaneous π phase pulse at evenly spaced intervals. This preserves their relative phase at the pulse itself while reversing the relative sign between the rails and bath. The Hamiltonian coupling remains fixed.

## Measured physical-channel probabilities

The 49 simulated baths have different fitted return times. Each site was observed for **one additional return interval** after its successful catch. At each site all pulse counts had the same observation time. The table averages site-level probabilities with equal weights.

| π pulses during the hold | Loss at the end of the hold | Largest loss during the hold, averaged across sites | Entanglement infidelity given final rail occupancy |
| ---: | ---: | ---: | ---: |
| 0 | 6.565% | 38.222% | 1.334% |
| 2 | 34.349% | 44.180% | 18.249% |
| 8 | 23.388% | 33.627% | 8.600% |
| 32 | 2.410% | 4.496% | 0.566% |
| 128 | 0.153% | 0.298% | 0.365% |

Pulse spacing matters. With two pulses, the final loss exceeds the free-evolution value. With 128, both loss and encoded-state distortion fall. After 128 pulses, we also followed the *same joint rail–bath state* through a second interval. Continuing the pulse train gave a mean peak loss of **0.279%** in that interval. Stopping it gave **38.401%**. [Summary](measured/summary.json) · [Per-site results](measured/per-bath.csv) · [Release test](measured/release.csv)

## What the calculation represents

The initial logical input is half of a maximally entangled state, so its marginal on the two rails is maximally mixed. The successful occupancy check projects onto the rails and the fit supplies the catch-frame correction. We propagate the conditional state and its coherence with the near field through the hold, without subsequent measurements or bath resets. For each pulse count, the fitted channel supplies a known final frame correction. Conditional entanglement infidelity measures the remaining logical-state distortion after that correction. Endpoint loss measures the probability of an empty rail pair at the end, including amplitude still in the bath. Peak loss is the maximum over **257 equally spaced** time samples per interval for each site.

This is the paper’s first-attachment, cold single-excitation model with zero direct rail dephasing. Vacuum absorbs all Lindblad jumps. The no-jump rail block therefore gives the exact survival amplitudes of that model. We checked all 49 catch probabilities against the published table and independently checked the full Lindblad density-matrix evolution at every tested pulse count on the first bath. The calculation assumes noiseless instantaneous control and the availability of the fitted frame corrections. Finite-width and imperfect pulses, persistence until a hardware cooldown, and a surface-code memory running the pulse sequence are separate tests. The Stim logical-error and decoder timing tables use the check schedules only. [Upstream model](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py)

## Replay

With NumPy, SciPy, QuTiP and pandas installed, run the following from the project root. Use a new output directory.

```sh
python experiments/bath-pulse-hold-02/run.py PATH_TO_PINNED_RQL_CHECKOUT results/pulse-hold-replay
```

The script checks the source checkout against commit `94572516042297d6b6018fe76eb2f2b547e3c46f`. [Implementation](run.py)
