# Sakidja Table 5 fresh-shot replay

This replay uses the first rail attachment (`SQB_ATTACH=one`), zero direct rail dephasing, clock period 6, and distance-7 rotated Z memory. It compares the location-matched, frame-calibrated clock with the fitted balanced check. The author’s simulator generates seven syndrome rounds and both matching decoders receive the simulated erasure flags. The clock uses the true bath parameters for its frame and matching weights. The fitted arm uses the published fitted parameters for its frame and matching weights. [Upstream model and decoder](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py) · [Published surface-code results](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/table3_surface_code.csv)

| Arm | Published failures | Fresh replay failures | Fresh logical error rate |
| --- | ---: | ---: | ---: |
| Frame-calibrated clock with location-matched decoding | 3,407 / 20,000 | 3,345 / 20,000 | 16.725% |
| ML then fit with balanced check | 0 / 20,000 | 4 / 20,000 | 0.020% |

Reusing the published shot seeds locally gave 3,438 and 2 failures, respectively. The source repository lists unpinned dependencies. Stim documents that seeded output depends on its version and can vary across machines with different SIMD widths. [Fresh counts and seed provenance](result.json) · [Stim sampler reference](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/python_api_reference_vDev.md#stim.Circuit.compile_detector_sampler)

## What the initialization does

Sakidja describes calibrating each qubit at cooldown as a power-on self-test. Measurements on prepared calibration states estimate how the excitation moves between that qubit’s rails and its surroundings. The controller can then schedule a check near a predicted return and compensate the rotation expected on return. The check measures whether the excitation occupies the rail pair while preserving the distinction between the two logical values. A flagged loss gives the surface code a location to use during recovery. Both replay arms continue through all seven rounds and score every shot. The replay starts from the author’s saved calibration outputs.

The paper also proposes starting a train of phase flips after the predicted return to suppress renewed exchange. Testing that proposal requires simulating the driven rail and bath together, including the state left in the bath between checks. The numerical result above measures scheduled checks and frame correction. [Sakidja’s model and scope](https://github.com/sakidja/RQL_Quantum_bath/tree/94572516042297d6b6018fe76eb2f2b547e3c46f#what-the-paper-does-not-cover)

## Replay inputs

Each arm uses the same five published bath layouts of 49 data qubits, with 4,000 fresh Stim shots per layout. The author samples each layout with replacement from 49 unseen simulated baths. The driver preserves the author’s layout seeds and uses disjoint fresh Stim seeds for the two arms. The result file records both sets of shot seeds and all layout assignments in data-qubit order.

The driver reads the published physical channels and selected check times. At each saved time, it recomputes the frame and physical channel from the published true and fitted parameters. The reconstructed probabilities agree with the CSV to machine precision for all 98 arm-bath rows. For the fitted arm it also constructs the decoder’s prior channel from the published fitted parameters using the author’s `round_channel` function. The driver starts after calibration, training, fitting and check-time selection. It uses the author’s `logical_errors` function for circuit generation, sampling and decoding. [Published per-bath parameters](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/raw_parameters_per_bath.csv) · [Published per-bath channels](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/raw_round_channels_per_bath.csv)

The mean physical erasure probability across the 49 unseen baths is 31.62% for the clock and 7.73% for the fitted balanced check. The clock checks at time 6. The fitted check times range from 6.36 to 15 and average 11.17. Both published bath channels have zero unflagged X and Y probabilities, so this Z-memory comparison primarily probes the change in erasure probability. The simulation adds the same 0.1% X and Z check-error probabilities and 0.1% measurement and reset flip probabilities in both arms. Its bath model starts each round in the bath ground state and maps each chosen time to a fixed one-round channel. [Published channel averages](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/results_v3_final8_one_gamma_sys_0.0/table2_round_channels.csv) · [Upstream model assumptions](https://github.com/sakidja/RQL_Quantum_bath/tree/94572516042297d6b6018fe76eb2f2b547e3c46f#what-the-paper-does-not-cover)

## Run locally

From this repository’s root, with the dependency versions in [result.json](result.json), run the following in PowerShell.

```powershell
$upstream = Join-Path $env:TEMP 'sakidja-rql-quantum-bath'
git clone https://github.com/sakidja/RQL_Quantum_bath.git $upstream
git -C $upstream checkout 94572516042297d6b6018fe76eb2f2b547e3c46f
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python experiments/sakidja-table5-replay-01/replay.py $upstream
```
