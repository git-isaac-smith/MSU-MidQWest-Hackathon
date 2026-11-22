# Does a bath-derived prior improve correlated matching?

This confirmation uses the existing decoder and the synthetic split-rail calibration from Iteration 03. The physical circuit is the project's provisional `simple_uniform` model. The bath supplies a hypothesis about the relative X, Y and Z weights at round-start data idles. The mapping of rail axes to data-qubit axes is an assumption fixed before sampling.

## Primary comparison

Compare correlated PyMatching with the fixed-clock bath weights against correlated PyMatching with equal X, Y and Z weights. Both use the same procedure to expand the model's idle errors into independent Pauli mechanisms. The only configuration input that differs is the axis mixture. All methods decode every shot from the original circuit. Only the scorer reads the sampled observable. The decoder receives no erasure flags, bath trajectory, or adjusted physical clock.

Use the existing 24 scenarios with equal weight. Distances are 3, 5 and 7, rounds equal distance, X and Z memories, and the four physical error parameters are 0.001, 0.003, 0.005 and 0.01. Use 125,000 new shots per scenario, for 3,000,000 paired shots, with root seed 42291137. The previous experiments suggested about 0.51% discordance and an effect around 0.012 percentage points. Fix the sample count without intermediate decisions.

Report candidate minus control logical error percentage, the paired 95% normal interval stratified by scenario, and the numbers of recovered and newly failed shots. A primary interval wholly below zero supports a benefit on this grid. An interval spanning zero leaves the direction unresolved. Comparisons to native matching, ordinary matching, the alternative check-time prior, and individual distances and bases are secondary. Report host batch timing as descriptive from one warmed full batch per scenario, with setup and individual latency recorded separately. This study has no prespecified speed claim.

## Reproduce

The reused Iteration 03 runner writes its historical native-matcher contrast into `run/manifest.json`. That field describes the original study. This confirmation specifies the expanded-uniform contrast above before reading any outcomes. The runner records both comparisons from the same shots.

From the repository root, run the following command with the project environment active and a fresh output path.

```sh
python -m bascd.bath_transfer --profile experiments/iteration-03/bath-calibration/bath-profile.json --out results/my-bath-confirmation --seed 42291137 --shots 125000 --repeats 1 --latency-shots 32
```
