# Single-record learned decoding

The scalar API reproduced the frozen learned policy's routing decisions and predicted bits on every one of 6,144 fresh detector records. Median call time fell from 61.15 to 11.45 µs at distance 5 and from 46.70 to 14.20 µs at distance 7, a 5.34× and 3.29× improvement over calling the batch API with one row.

| Distance | Bytes-to-int API | Median (µs) | p95 (µs) | p99 (µs) |
| --- | --- | ---: | ---: | ---: |
| 5 | Ordinary PyMatching | 8.20 | 15.70 | 24.73 |
| 5 | Correlated PyMatching | 14.40 | 31.70 | 48.26 |
| 5 | Learned batch, one row | 61.15 | 99.84 | 170.80 |
| 5 | Learned scalar | **11.45** | **30.60** | **49.07** |
| 7 | Ordinary PyMatching | 17.10 | 36.35 | 57.48 |
| 7 | Correlated PyMatching | 39.20 | 81.99 | 126.23 |
| 7 | Learned batch, one row | 46.70 | 139.59 | 221.62 |
| 7 | Learned scalar | **14.20** | **71.00** | **120.07** |

At distance 7, the scalar learned median was 17% lower than ordinary matching. Its p95 and p99 were higher by 34.65 and 62.58 µs. The gate selected full correlated matching on 171 of 3,072 distance-5 records and 169 of 3,072 distance-7 records.

These measurements start with one complete packed detector record in host memory and end when the integer logical bit is available. All four entry points accept the same `bytes` input, and the direct ordinary and correlated controls use the same packed PyMatching interface as the learned scalar adapter. For each distance, the run used three independent timing seeds of 1,024 shots each, 128 separate warm-up records, and randomized method order for each timed record. The circuit, policy and environment details are recorded in the [frozen run protocol](../compute-frontier-02/protocol.json) and [latency measurements](latency.json).

## Use the scalar decoder

```python
from bascd.scalar_routing import ScalarLearnedDecoder

decoder = ScalarLearnedDecoder.from_case("experiments/compute-frontier-02/d7")
prediction = decoder.decode(packed_record)
```

`packed_record` is a complete little-endian detector record as `bytes`. `decode` returns integer 0 or 1. `route` returns whether the policy selects correlated matching.

## Reproduce the timing

```powershell
python experiments/compute-scalar-01/measure.py --run experiments/compute-frontier-02 --out results/scalar-replay-latency.json
```
