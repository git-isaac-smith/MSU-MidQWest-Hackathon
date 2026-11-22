# Specialist iteration budget

Provisional simple_uniform circuit noise, rotated Z memory, d=7, rounds=7, p=.005.

Development selected **minimum-8-08** under the frozen rule. Confirmation used 180,000 fresh paired shots. Warm serial timing used 9 paired batches of 8,192 shots. Logical error is measured per complete memory shot.

| Method | Failures | Error, % | Recovered vs correlated | Introduced vs correlated | Paired p | Median batch µs/shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| product-12 | 999 | 0.5550 | 295 | 169 | 5.284e-09 | 83.764 |
| minimum-8-08 | 1,044 | 0.5800 | 283 | 202 | 0.0002723 | 51.018 |
| correlated | 1,125 | 0.6250 | 0 | 0 | 1 | 33.222 |
| ordinary | 1,857 | 1.0317 | 400 | 1132 | 5.194e-81 | 12.137 |

The selected budget reduced failures under the prespecified paired criterion.
Other accuracy comparisons and all timing results are descriptive.

## Paired comparison with the original hybrid budget

The original hybrid uses up to 12 product-sum iterations on disagreements. Serial timing includes both matching votes, selection and all belief calls.

| Test seed | Original failures | Selected failures | Recovered vs original | Introduced vs original |
| ---: | ---: | ---: | ---: | ---: |
| 92640100 | 335 | 349 | 23 | 37 |
| 92640101 | 340 | 358 | 20 | 38 |
| 92640102 | 324 | 337 | 19 | 32 |

The original hybrid took 1.627× the selected budget’s time at the median paired batch ratio.

[Protocol](protocol.json) · [Development ranking](development.csv) · [Frozen selection](frozen.json) · [Confirmation accuracy](confirmation.csv) · [Per-seed accuracy](confirmation-per-seed.csv) · [Confirmation timing repetitions](confirmation-timing.csv)
