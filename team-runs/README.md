# Run a team experiment from the browser

Join the host's Live Share session using **Join from the Web** and sign in for editing. All participants edit the same project. The host runs saved requests one at a time in an isolated Linux environment.

1. Open `browser-check.request.json` and copy its contents. Create a new file in this folder with a fresh name such as `alex-01.request.json`. Paste the JSON, set your parameters, then save to submit.
2. Open the matching `alex-01.result.json` to see the status, failures for each decoder, or an error. The `alex-01.out` folder holds the full experiment results.
3. Use a fresh name for each run. The runner picks up saved edits to the project's Python code when each new experiment starts. Each teammate can submit their own file independently.

For an Iteration 02 sweep, save `{"mode":"sweep","shots":10000,"seed":2026}`. For a random-search campaign, save `{"mode":"random","budget":2,"shots":1000,"seed":2026}`. Both run the 24-scenario grid. Each job has a 15-minute limit.

The desktop Live Share session also lists **MidQWest Team Lab** under **Shared Servers**. Open it, use the login token from the host, then open `Team Benchmark.ipynb` for collaborative notebook editing and execution.
