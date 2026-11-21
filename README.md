# MSU-MidQWest-Hackathon
Repository for Missouri State University's Quantum Bears MidQWest Hackathon Project - Northrop Grumman Track

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

The team's current contribution is the paired benchmark and scoring code. The project outline proposes a small ensemble as the first original decoder experiment and a separate bath-informed erasure study. Comparing the ensemble requires held-out shots. Evaluating the erasure proposal requires a noise model that generates observable heralds.
