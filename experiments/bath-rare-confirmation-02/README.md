# Fresh confirmation of the distance-7 fitted bath result

The fitted balanced check failed on **74 of 500,000 fresh shots**, a logical error rate of **0.01480%** with a 95% Wilson interval of **0.01179% to 0.01858%**. The original stress run observed 4 failures in 20,000 shots under the same configuration. [Fresh counts, layouts, and seed provenance](result.json) · [Original counts](../bath-background-stress-01/result.json)

| Layout | Fresh failures / shots | Logical error rate | 95% Wilson interval |
| ---: | ---: | ---: | ---: |
| 1 | 6 / 100,000 | 0.00600% | 0.00275%–0.01309% |
| 2 | 21 / 100,000 | 0.02100% | 0.01374%–0.03210% |
| 3 | 15 / 100,000 | 0.01500% | 0.00909%–0.02475% |
| 4 | 6 / 100,000 | 0.00600% | 0.00275%–0.01309% |
| 5 | 26 / 100,000 | 0.02600% | 0.01774%–0.03809% |

This experiment uses Sakidja’s rotated distance-7 Z-memory circuit with seven syndrome rounds, the first rail attachment, zero direct rail dephasing, nominal clock period 6, and the fitted balanced-check arm at the native `P_CHECK=0.001`. The sampled circuit uses the published physical channels. The decoder prior uses the channels reconstructed from the fitted bath parameters. Across the 49 unseen baths, the fitted check time averages 11.1735 and the physical erasure probability averages 7.7282%. [Pinned upstream circuit and decoder](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py#L389-L510) · [Original stress experiment](../bath-background-stress-01/README.md)

The run fixes the five original bath-to-qubit layouts and draws 100,000 fresh shots per layout in twenty 5,000-shot chunks. All 100 Stim seeds are deterministic and distinct. Each layout first replays its original 4,000-shot seed and matches the saved logical failures and the herald and syndrome event totals for both outcomes. The confirmation starts with a fresh decoder for each layout and counts only fresh shots. The 95% Wilson intervals describe Monte Carlo precision conditional on these layouts.

The independent serial measurement on the original 20,000-shot batch was 944.14 µs per shot, or 134.88 µs per round. That measurement times `ErasureDecoder.decode(detectors)` after warm-up. The confirmation uses up to five workers for accuracy sampling and reports no inference timing. [Original serial benchmark](../bath-background-stress-01/timing/README.md)

## Reproduce

Run from the repository root in PowerShell with the package versions in the result. The five workers use one thread each for numerical libraries. Keep the recorded chunk size when reproducing the Stim draws.

```powershell
$upstream = Join-Path $env:TEMP 'sakidja-rql-quantum-bath'
git clone https://github.com/sakidja/RQL_Quantum_bath.git $upstream
git -C $upstream checkout 94572516042297d6b6018fe76eb2f2b547e3c46f
python experiments/bath-rare-confirmation-02/run.py $upstream --output bath-rare-replay.json
```
