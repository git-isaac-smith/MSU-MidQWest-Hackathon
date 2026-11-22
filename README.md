# MSU-MidQWest-Hackathon
Repository for Missouri State University's Quantum Bears MidQWest Hackathon Project - Northrop Grumman Track

## Quick start

Install Git and Python 3.11, then clone the repository.

```sh
git clone https://github.com/git-isaac-smith/MSU-MidQWest-Hackathon.git
cd MSU-MidQWest-Hackathon
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m bascd.demo
```

**macOS and Linux**

```sh
python3.11 -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m bascd.demo
```

The demo compares ordinary minimum-weight perfect matching (MWPM), correlated matching, and selective correlated matching on the same 10,000 Stim syndromes from a distance-7 Z memory with seven rounds and `p=0.005`. It prints logical failures, error rates, batch decoding time per shot, and paired recoveries and regressions against ordinary MWPM. The selective decoder calibrates its activity cutoff on a separate unlabeled sample.

[Presentation deck](output/presentations/MidQWest%202026%20-%20Selective%20Correlated%20Matching%20%28presentation%20cut%29.pptx) · [Circuit-level benchmark](Circuit%20decoder%20speed%20and%20accuracy.md) · [Earlier FFT/CNN experiments](new-approach/README.md)

## Challenge and submission

The organizer brief asks teams to implement and explain one or more decoder methods. The submission includes decoder code and test functions that calculate decoder metrics in a public Git repository, plus a roughly 10-minute presentation or whiteboard talk. Judges consider presentation quality, algorithm novelty, parallelizability, code quality, and the number of algorithms implemented. Runtime matters alongside the team's ability to explain the algorithm and its tradeoffs. The brief recommends Stim for syndrome generation. [Organizer's suggested surface-code decoding review](https://arxiv.org/abs/2307.14989v4)

The supplied brief does not specify exact noise channels, an input format, or a scoring formula. The current sweep uses provisional `simple_uniform` noise. The [Iteration 03 plan](Iteration%2002.md#iteration-03--structured-quantum-bath-decoder) covers the team's bath-aware decoder proposal, parallel scaling, and submission.

The four Quantum Bears modules motivate a test of bath-derived matching weights on fixed Stim records and a separate simulation of measured erasure flags. The [bath confirmation study](experiments/bath-confirmation-01/README.md) explains what information each decoder receives and which comparison to show the judges.

## Decoder scaling

The [logical error and decoder runtime page](experiments/qubit-scaling-01/index.html) compares the circuit-level decoders on both memory bases through distance 31. It links to the five-round FFT/CNN comparison and the single-round heralded bath experiment. Each workload reports its own paired logical failures, shot count and batch inference time.

To reproduce the fixed-circuit study in a new directory, install the optional benchmark backends and run the following commands from the repository root.

```sh
python -m pip install -e ".[hypergraph,scaling]"
python -m bascd.scale_study prepare --out results/my-scaling
python -m bascd.scale_study run --out results/my-scaling
python -m bascd.scale_report results/my-scaling
```

## Run the distance-based hybrid decoder

The new compute study trains a detector-only routing policy and tests belief matching as a referee when two matchers disagree. It includes frozen distance-5 and distance-7 models, paired accuracy tests, single-shot latency, and throughput with 1, 2, 4 and 8 worker processes. [Results, replay and deployment](experiments/compute-frontier-02/README.md)

The circuit-level speed and accuracy study compares ordinary and correlated PyMatching, weighted Union-Find, posterior lookup, selective correlated matching and two temporal prefilters. It includes paired held-out tests, offline batch time per round, single-record latency, finite-distance crossings and stress cases. [Benchmark results and deployment](Circuit%20decoder%20speed%20and%20accuracy.md)

The distance-based hybrid decoder uses a logical posterior lookup at distance 3. At distances 5 and 7, the decoder runs belief matching and a five-member bath-initialized matching ensemble. A bounded search resolves disagreements. The team locked the decoding rule for each distance before sampling the final test. The decoder accepts detector events alone. [Held-out results, installation, and replay](Iteration%2004.md) · [Saved decoder policy](experiments/iteration-04/accuracy/frozen-policy.json)

### Generate and verify Stim samples

With the `hypergraph` extra installed, run these commands from the repository root. The first command generates rotated X- and Z-memory circuits, writes each circuit as a `.stim` file, and samples 10,000 shots per scenario. It writes detector bits, observable flips and logical predictions from standard correlated matching, bath-weighted correlated matching, and the distance-based hybrid decoder to separate Stim `b8` files. The second command re-creates the seeded samples and independently recounts failures and paired outcomes.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m bascd.stim_samples --policy experiments/iteration-04/accuracy/frozen-policy.json --out results/stim-shots --shots 10000 --seed 954130
& $python scripts/verify_stim_samples.py results/stim-shots --resample
& $python scripts/display_stim_results.py results/stim-shots
```

The last command displays median and best observed logical success alongside median batch decoding time. Choose a fresh output directory for each sampling run. The manifest records the Stim version, circuit settings, per-case seeds and sampling call size. Exact seeded reproduction depends on compatible Stim versions and CPU instructions. The saved files can also be scored directly. [Decoder performance table](Iteration%2004.md#fresh-stim-shot-archive) · [Example Stim circuit](experiments/iteration-04/stim-shot-archive/cases/d3-z-p0.005/circuit.stim) · [Per-scenario statistics](experiments/iteration-04/stim-shot-archive/stats.csv)

## Run the surface-code baseline

From PowerShell at the repository root, create a Python 3.11 environment and install the package.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --output-dir results/smoke
```

In the VS Code Live Share session, run the last command in a shared host terminal. The package entry point `python -m bascd` accepts the same options.

The smoke run uses distance 3, three rounds, and 1,000 shots per basis at `p=0.001`. The X and Z memories use seeds 101 and 102. Each basis supplies identical detector events and a common decomposed error model to ordinary and correlated minimum-weight perfect matching (MWPM). Stim supplies the sampled logical flips to the evaluator. A shot fails if any predicted logical flip differs from the sampled flips.

The exploratory `simple_uniform` model applies depolarizing noise after Clifford gates and to data qubits before each round. It also introduces flips after resets and before measurements. All four channels use probability `p`. The organizer's noise channels, decoder interface, and timing rubric are open questions.

The CLI prints JSON. With `--output-dir`, it also writes `benchmark.json` with the four paired outcomes for each basis and `benchmark.csv` with one row per decoder and basis. Use a new directory to retain each run. Rows include the circuit hash, seed, versions, host information, failure rate, Wilson 95% interval, and setup, sampling, and decoding times. Batch microseconds per shot divides one timed decode call by the shot count. Measuring individual request latency requires a separate experiment.

For another case, set `--basis x` or `--basis z` and adjust `--distance`, `--rounds`, `--p`, `--shots`, and `--seed`. Rounds default to distance. For seeded replay, match the recorded software versions, CPU instruction set, and sampling batch size. [Stim simulator](https://github.com/quantumlib/Stim) · [PyMatching decoder](https://github.com/oscarhiggott/PyMatching)

## Iteration 02 experiments

The project's main novelty is a structured-bath approach proposed by Dr. Ridwan Sakidja. Fit the bath to time-domain calibration traces, use the inferred dynamics to schedule a dual-rail erasure check near an excitation return, then test whether repeated common π-phase pulses suppress subsequent exchange. The bath-aware decoder can use observed erasure flags and calibrated weights in the research model. The fixed-noise Stim sweep remains the reference benchmark. [Sakidja's Zenodo record](https://zenodo.org/records/24326452)

### Initialize the structured bath

```powershell
.\.venv\Scripts\python.exe -m bascd.bath_initialize --out results/bath-demo
.\.venv\Scripts\python.exe -m bascd.bath_initialize --traces path/to/calibration.npz --out results/bath-fit
```

The initializer fits a dual-rail model to calibration traces before scheduling a check. It exports the selected time, its frame correction, and the predicted erasure and Pauli probabilities in `bath-profile.json`. Use `--attachment one` if rail R couples to both first-layer nodes, or `--attachment split` if each rail couples to its own node. Use `--initial path/to/estimate.json` to refine an existing six-parameter estimate. The default run generates a reproducible synthetic trace. Input format, model assumptions, and the verified result are in the [bath initialization report](Iteration%2003.md#structured-bath-initialization).

The existing random-search pipeline provides a control for an optional offline generative-AI policy comparison. A deployed policy selects and compiles a PyMatching decoder for each code distance. Inference uses only detector events in the fixed-noise benchmark.

### Run the bath-initialized decoder

Use an exported bath profile to initialize ordinary and correlated matching for the provisional surface-code circuits. The transfer experiment compares the fitted Pauli-axis weights with native and equivalent expanded uniform weights. The second command tests an idealized Z memory in which the decoder can read a per-site erasure flag. Both commands require a new output destination.

```powershell
$python = ".\.venv\Scripts\python.exe"
$profile = "experiments/iteration-03/bath-calibration/bath-profile.json"
& $python -m bascd.bath_transfer --profile $profile --out results/transfer --seed 602711
& $python -m bascd.bath_erasure --profile $profile --output results/heralds/result.json --distance 3 --shots 20000 --seed 31
& $python scripts/plot_bath_results.py results/transfer/summary.json results/heralds/result.json results/transfer/comparison.png
```

The bath prior currently maps onto the verified pre-round data-qubit idle locations in Stim's generated rotated memories. The independent-replay command, decoder API, and limits of the evidence are in the [bath decoder report](Iteration%2003.md#bath-initialized-surface-code-decoding).

## Adaptive decoder for the fixed-noise challenge

The adaptive decoder chooses ordinary or correlated MWPM from the incoming syndrome. The learned lookup table uses either detection-event count or event count plus the number of adjacent occupied measurement rounds. Calibration groups paired ordinary and correlated predictions by feature bucket. The rule prefers buckets with more recoveries than regressions and limits the fraction routed to correlated matching. Time-domain bath inference motivates the use of temporal features. The fixed-noise decoder receives detector events and compiles against the judge circuit's detector error model.

See the [Iteration 03 report](Iteration%2003.md) for the experiment protocol and results.

```powershell
.\.venv\Scripts\python.exe -m bascd.adaptive_experiment --out results/adaptive-replay
.\.venv\Scripts\python.exe -m bascd.adaptive_experiment --circuit path/to/judge.stim --out results/judge-adaptive
```

Run the second command with an organizer-supplied circuit when available. The experiment trains on one seed stream, selects the lowest observed validation error rate under illustrative batch-runtime caps of 1.5× and 2× ordinary matching, then freezes the policies before sampling a third stream for test. Both reference decoders are eligible at selection. All compared decoders receive the same samples within each split. Batch timing includes feature calculation, routing, matching, and restoring shot order. Test output also reports single-shot latency and per-scenario Wilson intervals.

For deployment, load a per-circuit policy from `frozen-policies.json` with `RoutingPolicy.from_dict`. Build `AdaptiveDecoder(dem, policy)` once, then call `decode_batch(detectors)` or `decode_one(detectors)`. Build `dem` with `circuit.detector_error_model(decompose_errors=True)`. Use the exact calibrated model. The decoder verifies its fingerprint at construction. Independent chunks can use separate compiled instances and be concatenated in shot order.

### Reproduce the Iteration 02 baselines

Run the commands below from the repository root after installing the development environment above. Each sweep or campaign starts in a new directory. The recorded Iteration 02 runs are under `experiments/iteration-02/`. See the [results and Iteration 03 plan](Iteration%2002.md).

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m bascd.experiment sweep --out results/replay-sweep --shots 10000 --seed 2026 --repeats 3 --latency-shots 128
& $python -m bascd.experiment search --out results/replay-random --strategy random --budget 8 --shots 10000 --seed 2026 --search-seed 17 --repeats 3
& $python scripts/plot_search.py results/replay-random
```

The sweep has 24 scenarios spanning distances 3, 5, 7, four values of `p`, and X and Z memories with rounds equal to distance. Each scenario samples once. Both decoders use that batch, its separate observable labels for scoring, and the same decomposed detector error model. Every row gives the circuit and sample fingerprints. A run manifest records all four provisional noise channels, dependency versions, source hashes, host, command, and seeds or root seed. Replay with the recorded versions, CPU instruction set, and shot count. Wall times depend on machine load.

The logical error figure includes individual Wilson 95% intervals and retains zero-failure observations. For greater precision, set a larger fixed shot count, use a fresh seed, and restrict the new sweep to the relevant scenarios. For example, `--distances 7 --probabilities 0.001 --bases z --shots 100000 --seed 2027` runs the rare-event case independently. Choosing the follow-up size in advance preserves the interpretation of its interval.

Decoder setup includes graph construction, weight edits, and one untimed-for-inference warm-up to include lazy initialization in setup. Timed batch calls use the full syndrome matrix. The runtime plot shows median batch time divided by batch size and separately shows the 95th percentile of 128 individually timed decoder calls. Scoring and sample generation run outside both inference timers. The paired CSV breaks out each decoder's exclusive failures.

### Decoder policy

A policy supplies an entry for each of distances 3, 5, and 7. Each entry chooses ordinary MWPM, native correlated MWPM, or ordinary MWPM with adjusted edge weights. Native correlation uses PyMatching's correlation option at both construction and inference.

The weighted decoder multiplies the ordinary graph's log-odds weights by a temporal scale for edges joining the same spatial coordinates across rounds, and by a boundary scale for edges ending at the virtual boundary. Spatial and diagonal space-time edges retain their calibrated weights and anchor the relative scale. Each multiplier is 0.75, 1, or 1.25. Weighted mode requires at least one change. Coordinates, edge endpoints, logical fault IDs, sampled data, and the physical Stim circuit retain their original meaning. The default graph's edge probability metadata stays available for inspection. This benchmark samples noise exclusively through Stim. [PyMatching implementation](https://github.com/oscarhiggott/PyMatching/blob/master/src/pymatching/matching.py)

These choices give ten implementations at each distance and 1,000 distinct three-distance policies. The interface excludes editable correlated edges because the ordinary edge representation omits the correlation metadata. Boundary topology, clustering priorities, and conditional fallbacks have no implemented control in this search space.

### Proposals, budgets, and records

An offline proposer can produce a JSON file such as the following. This is an interface example.

```json
{
  "by_distance": {
    "3": {"decoder_type": "mwpm"},
    "5": {"decoder_type": "weighted_mwpm", "temporal_scale": 0.75, "boundary_scale": 1},
    "7": {"decoder_type": "correlated_mwpm"}
  }
}
```

```powershell
& $python -m bascd.experiment search --out results/guided --strategy guided --budget 8 --shots 10000 --seed 2026 --repeats 3
& $python -m bascd.experiment validate proposal.json
& $python -m bascd.experiment submit --out results/guided proposal.json
```

Both strategies use the same proposal validator, compiler, measurement code, and scoring rule. The seeded control samples uniformly without replacement. A submitted duplicate, malformed proposal, or decoder failure consumes one slot and retains its diagnostic. Standalone validation checks syntax and configuration without exposing a score. The eighth submission in the example closes proposals and starts validation. Use `resume --out DIRECTORY` to recover an interrupted campaign. Interrupted training consumes its reserved slot. A process lock protects the budget and the JSON ledger is replaced atomically.

During search, the proposer reads `feedback.json` and its own `measurements/NNN-train.json` files. Validation sampling begins after all proposal slots are spent. Two fixed reference matchers also run on that validation split, outside the proposal budget, so each candidate has paired counts against both. Train, validation, exploratory sweep, and frozen final-test seeds occupy disjoint namespaces. The sampler reserves the final namespace for Iteration 03.

Each scenario has equal shot count and contributes equally to the search objectives. Each trial records per-scenario Wilson intervals. Its aggregate LER has a conservative 95% Hoeffding interval for independent, potentially heterogeneous shots. The empirical Pareto frontier minimizes validation LER and batch inference time among successful candidates. Equal points are retained. The full records are in `history.json`, `history.csv`, the per-trial measurement files, and `pareto.json`. The latter lists trial numbers. The validation reference rows are in `validation-references.json`. [Proposal schema](src/bascd/schemas/proposal.json) · [Feedback schema](src/bascd/schemas/feedback.json)
