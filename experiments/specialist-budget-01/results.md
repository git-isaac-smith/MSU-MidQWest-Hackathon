# Specialist iteration budget

Provisional simple_uniform circuit noise, rotated Z memory, d=7, rounds=7, p=.005.

Development selected **product-4** under the frozen rule. Confirmation used 180,000 fresh paired shots. Warm serial timing used 9 paired batches of 8,192 shots. Logical error is measured per complete memory shot.

| Method | Failures | Error, % | Recovered vs correlated | Introduced vs correlated | Paired p | Median batch µs/shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| product-12 | 971 | 0.5394 | 270 | 171 | 2.795e-06 | 76.559 |
| product-4 | 999 | 0.5550 | 284 | 213 | 0.001663 | 50.513 |
| correlated | 1,070 | 0.5944 | 0 | 0 | 1 | 31.519 |
| ordinary | 1,754 | 0.9744 | 377 | 1061 | 1.91e-75 | 11.434 |

The selected budget reduced failures under the prespecified paired criterion.
Other accuracy comparisons and all timing results are descriptive.

## Paired comparison with the original hybrid budget

The original hybrid uses up to 12 product-sum iterations on disagreements. Serial timing includes both matching votes, selection and all belief calls.

| Test seed | Original failures | Selected failures | Recovered vs original | Introduced vs original |
| ---: | ---: | ---: | ---: | ---: |
| 92500100 | 336 | 347 | 23 | 34 |
| 92500101 | 332 | 334 | 15 | 17 |
| 92500102 | 303 | 318 | 14 | 29 |

The original hybrid took 1.516× the selected budget’s time at the median paired batch ratio.

[Protocol](protocol.json) · [Development ranking](development.csv) · [Frozen selection](frozen.json) · [Confirmation accuracy](confirmation.csv) · [Per-seed accuracy](confirmation-per-seed.csv) · [Confirmation timing repetitions](confirmation-timing.csv)
