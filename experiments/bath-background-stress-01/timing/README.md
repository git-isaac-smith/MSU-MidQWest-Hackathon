# Bath stress decoder timing

Each point replays the original 20,000 shots in their five 4,000-shot layouts. The runner warms the upstream erasure decoder on 64 separate shots for each layout, then measures one serial call to `ErasureDecoder.decode(detectors)`. The reported batch time sums those five wall-clock durations and divides by 20,000. The per-round value divides the batch time by seven. The timer excludes circuit generation, sampling, decoder construction, warm-up, and scoring. The timing run used an Intel Core i7-13700HX under Windows.

The runner checks each layout against the original failure count and the original totals of herald and syndrome detector events for successful and failed shots. It also checks the pooled failure count. [Timing results](result.json) · [Original stress experiment](../result.json)

Run from the repository root with Python and the package versions recorded in the original result. Use a checkout of the upstream repository at the saved revision and choose a fresh output file for a new run.

```powershell
python experiments/bath-background-stress-01/timing/run.py $upstream --output bath-timing-replay.json
```
