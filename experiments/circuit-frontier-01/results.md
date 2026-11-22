# Circuit-level decoder results

Logical error rates are per complete memory experiment. Batch timing is measured in microseconds per shot. The per-round batch metric in the CSV divides that time by the number of rounds and describes offline throughput. Paired tests compare logical outcomes on the same detector shots.

## Primary held-out case

The primary hypothesis compares fast hybrid with ordinary MWPM on the Z memory at distance 3 and p = 0.005. Success requires fewer errors with two-sided exact McNemar p < 0.05 and lower median time for warmed, interleaved batches. Other decoders in the primary table are exploratory.

| Case | Decoder | Errors / shots | LER | Batch µs/shot | Fixed / added vs MWPM | Paired p |
|---|---|---:|---:|---:|---:|---:|
| primary-d3-z-p0.005-ordinary | Ordinary MWPM | 5,261 / 300,000 | 1.7537% | 0.470 | 0 / 0 | 1 |
| primary-d3-z-p0.005-ordinary | Correlated MWPM | 5,123 / 300,000 | 1.7077% | 1.594 | 1,029 / 891 | 0.00176 |
| primary-d3-z-p0.005-ordinary | Union-find | 5,334 / 300,000 | 1.7780% | 1.280 | 42 / 115 | 4.85e-09 |
| primary-d3-z-p0.005-ordinary | Fast hybrid | 4,455 / 300,000 | 1.4850% | 0.038 | 1,477 / 671 | 3.34e-69 |
| primary-d3-z-p0.005-ordinary | CSS routed | 5,147 / 300,000 | 1.7157% | 0.677 | 611 / 497 | 0.00068 |
| primary-d3-z-p0.005-ordinary | Temporal low-pass | 34,360 / 300,000 | 11.4533% | 1.454 | 1,837 / 30,936 | 0 |
| primary-d3-z-p0.005-ordinary | Local peel + MWPM | 5,261 / 300,000 | 1.7537% | 0.609 | 0 / 0 | 1 |

Fast hybrid met the joint primary criterion with 1,477 fixed, 671 added, p = 3.34e-69, and an MWPM-to-hybrid batch runtime ratio of 12.44×.

## Secondary held-out confirmations

The prespecified secondary comparisons evaluate CSS routed against ordinary MWPM at distances 5 and 7 in Z memory with p = 0.005. The Bonferroni-adjusted cutoff is p < 0.025 for each two-sided exact paired test. Joint success also requires a lower measured median batch time.

| Distance | Errors / shots, CSS vs MWPM | Change in LER | Fixed / added | Paired p | Batch µs/shot, CSS vs MWPM | MWPM / CSS time | Joint result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 2,057 vs 2,199 / 150,000 | -0.0947 pp | 295 / 153 | 1.85e-11 | 3.350 vs 3.381 | 1.01× | Met both criteria |
| 7 | 1,410 vs 1,515 / 150,000 | -0.0700 pp | 205 / 100 | 1.81e-09 | 9.337 vs 12.071 | 1.29× | Met both criteria |

## Exploratory wins

The table screens the remaining comparisons for lower logical error (two-sided paired p < 0.05) and lower measured batch time than ordinary MWPM. The screen uses unadjusted p-values.

| Study | Case | Decoder | LER vs MWPM | Speedup | Paired p |
|---|---|---|---:|---:|---:|
| sweep | sweep-d3-z-p0.003-ordinary | Fast hybrid | 0.4400% vs 0.6000% | 10.03× | 3.31e-06 |
| sweep | sweep-d3-x-p0.003-ordinary | Fast hybrid | 0.5350% vs 0.7050% | 8.36× | 3.33e-05 |
| sweep | sweep-d3-z-p0.005-ordinary | Fast hybrid | 1.4800% vs 1.7050% | 15.04× | 0.000312 |
| sweep | sweep-d3-x-p0.005-ordinary | Fast hybrid | 1.6400% vs 2.1750% | 13.83× | 2.61e-13 |
| sweep | sweep-d5-x-p0.005-ordinary | CSS routed | 1.5100% vs 1.6050% | 1.01× | 0.0319 |
| sweep | sweep-d7-z-p0.005-ordinary | CSS routed | 0.9350% vs 1.0300% | 1.26× | 0.00256 |
| sweep | sweep-d7-x-p0.005-ordinary | CSS routed | 1.2350% vs 1.3500% | 1.14× | 0.00177 |
| sweep | sweep-d3-z-p0.007-ordinary | Fast hybrid | 2.5950% vs 3.0900% | 14.51× | 1.52e-09 |
| sweep | sweep-d3-x-p0.007-ordinary | Fast hybrid | 2.9750% vs 3.8200% | 18.50× | 9.39e-20 |
| sweep | sweep-d5-x-p0.007-ordinary | CSS routed | 4.0050% vs 4.1950% | 1.09× | 0.00194 |
| sweep | sweep-d7-z-p0.007-ordinary | CSS routed | 3.3350% vs 3.4750% | 1.30× | 0.00842 |
| sweep | sweep-d7-x-p0.007-ordinary | CSS routed | 3.8700% vs 4.1150% | 1.30× | 1.12e-06 |
| sweep | sweep-d3-z-p0.01-ordinary | Fast hybrid | 5.3350% vs 6.0850% | 26.64× | 7.83e-12 |
| sweep | sweep-d3-x-p0.01-ordinary | Fast hybrid | 5.5800% vs 6.9000% | 17.08× | 8.55e-25 |
| sweep | sweep-d5-z-p0.01-ordinary | CSS routed | 8.1550% vs 8.3250% | 1.21× | 0.00685 |
| sweep | sweep-d7-z-p0.01-ordinary | CSS routed | 10.0950% vs 10.2700% | 1.35× | 0.0113 |
| sweep | sweep-d7-x-p0.01-ordinary | CSS routed | 12.1700% vs 12.3850% | 1.15× | 0.00815 |
| sweep | sweep-d3-z-p0.015-ordinary | Fast hybrid | 10.3400% vs 11.3450% | 25.01× | 3.34e-11 |
| sweep | sweep-d3-x-p0.015-ordinary | Fast hybrid | 10.7700% vs 12.5350% | 30.00× | 8.14e-26 |
| sweep | sweep-d3-z-p0.02-ordinary | Fast hybrid | 16.0800% vs 17.3950% | 39.24× | 2.89e-13 |
| sweep | sweep-d3-x-p0.02-ordinary | Fast hybrid | 17.0000% vs 19.2450% | 48.52× | 8.05e-27 |
| stress | stress-d3-z-p0.005-final_round_measurement | Fast hybrid | 1.5350% vs 1.8200% | 14.90× | 1.47e-06 |
| stress | stress-d3-x-p0.005-final_round_measurement | Fast hybrid | 1.4950% vs 1.9850% | 11.26× | 7.97e-13 |
| stress | stress-d3-z-p0.005-measurement_burst | Fast hybrid | 1.4550% vs 1.8800% | 13.68× | 4.07e-10 |
| stress | stress-d3-x-p0.005-measurement_burst | Fast hybrid | 1.4850% vs 2.3150% | 15.70× | 2.4e-25 |
| stress | stress-d3-z-p0.005-spatial_correlated | Fast hybrid | 5.4300% vs 7.8450% | 15.02× | 5.3e-58 |
| stress | stress-d3-x-p0.005-spatial_correlated | Fast hybrid | 5.2500% vs 7.8000% | 17.24× | 4.81e-64 |
| stress | stress-d5-x-p0.005-final_round_measurement | CSS routed | 1.4850% vs 1.6200% | 1.03× | 0.00244 |
| stress | stress-d5-z-p0.005-measurement_burst | CSS routed | 1.7500% vs 1.8650% | 1.14× | 0.00379 |
| stress | stress-d5-x-p0.005-measurement_burst | CSS routed | 1.9050% vs 2.0600% | 1.13× | 0.000244 |
| stress | stress-d7-z-p0.005-final_round_measurement | CSS routed | 0.8800% vs 0.9500% | 1.24× | 0.0243 |
| stress | stress-d7-x-p0.005-final_round_measurement | CSS routed | 1.1800% vs 1.3450% | 1.22× | 3.39e-06 |
| stress | stress-d7-z-p0.005-measurement_burst | CSS routed | 1.2050% vs 1.3250% | 1.36× | 6.96e-05 |
| stress | stress-d7-x-p0.005-measurement_burst | CSS routed | 1.4400% vs 1.5850% | 1.15× | 2.49e-05 |
| stress | stress-d7-z-p0.005-spatial_correlated | CSS routed | 6.0550% vs 6.2300% | 1.31× | 0.000103 |
| stress | stress-d7-x-p0.005-spatial_correlated | CSS routed | 7.3300% vs 7.4950% | 1.20× | 0.00853 |

## Stress cases

| Case | Decoder | LER | Batch µs/shot | Fixed / added vs MWPM |
|---|---|---:|---:|---:|
| stress-d3-z-p0.005-final_round_measurement | Ordinary MWPM | 1.8200% | 0.521 | 0 / 0 |
| stress-d3-z-p0.005-final_round_measurement | Correlated MWPM | 1.7850% | 1.667 | 66 / 59 |
| stress-d3-z-p0.005-final_round_measurement | Union-find | 1.8950% | 1.653 | 2 / 17 |
| stress-d3-z-p0.005-final_round_measurement | Fast hybrid | 1.5350% | 0.035 | 98 / 41 |
| stress-d3-z-p0.005-final_round_measurement | CSS routed | 1.7850% | 0.562 | 25 / 18 |
| stress-d3-z-p0.005-final_round_measurement | Temporal low-pass | 13.8250% | 1.782 | 158 / 2,559 |
| stress-d3-z-p0.005-final_round_measurement | Local peel + MWPM | 1.8200% | 0.663 | 0 / 0 |
| stress-d3-x-p0.005-final_round_measurement | Ordinary MWPM | 1.9850% | 0.707 | 0 / 0 |
| stress-d3-x-p0.005-final_round_measurement | Correlated MWPM | 1.9600% | 2.380 | 87 / 82 |
| stress-d3-x-p0.005-final_round_measurement | Union-find | 2.0100% | 1.654 | 5 / 10 |
| stress-d3-x-p0.005-final_round_measurement | Fast hybrid | 1.4950% | 0.063 | 145 / 47 |
| stress-d3-x-p0.005-final_round_measurement | CSS routed | 1.9650% | 0.762 | 29 / 25 |
| stress-d3-x-p0.005-final_round_measurement | Temporal low-pass | 15.2400% | 2.046 | 198 / 2,849 |
| stress-d3-x-p0.005-final_round_measurement | Local peel + MWPM | 1.9850% | 0.842 | 0 / 0 |
| stress-d3-z-p0.005-measurement_burst | Ordinary MWPM | 1.8800% | 0.499 | 0 / 0 |
| stress-d3-z-p0.005-measurement_burst | Correlated MWPM | 1.7350% | 1.521 | 81 / 52 |
| stress-d3-z-p0.005-measurement_burst | Union-find | 1.9200% | 1.359 | 7 / 15 |
| stress-d3-z-p0.005-measurement_burst | Fast hybrid | 1.4550% | 0.036 | 136 / 51 |
| stress-d3-z-p0.005-measurement_burst | CSS routed | 1.8600% | 0.607 | 29 / 25 |
| stress-d3-z-p0.005-measurement_burst | Temporal low-pass | 11.0900% | 1.481 | 124 / 1,966 |
| stress-d3-z-p0.005-measurement_burst | Local peel + MWPM | 1.8800% | 0.618 | 0 / 0 |
| stress-d3-x-p0.005-measurement_burst | Ordinary MWPM | 2.3150% | 0.543 | 0 / 0 |
| stress-d3-x-p0.005-measurement_burst | Correlated MWPM | 1.9700% | 1.558 | 111 / 42 |
| stress-d3-x-p0.005-measurement_burst | Union-find | 2.3500% | 1.379 | 18 / 25 |
| stress-d3-x-p0.005-measurement_burst | Fast hybrid | 1.4850% | 0.035 | 218 / 52 |
| stress-d3-x-p0.005-measurement_burst | CSS routed | 2.1650% | 0.663 | 49 / 19 |
| stress-d3-x-p0.005-measurement_burst | Temporal low-pass | 12.9700% | 1.503 | 192 / 2,323 |
| stress-d3-x-p0.005-measurement_burst | Local peel + MWPM | 2.3150% | 0.661 | 0 / 0 |
| stress-d3-z-p0.005-spatial_correlated | Ordinary MWPM | 7.8450% | 0.497 | 0 / 0 |
| stress-d3-z-p0.005-spatial_correlated | Correlated MWPM | 8.0650% | 1.367 | 77 / 121 |
| stress-d3-z-p0.005-spatial_correlated | Union-find | 7.8800% | 1.344 | 11 / 18 |
| stress-d3-z-p0.005-spatial_correlated | Fast hybrid | 5.4300% | 0.033 | 714 / 231 |
| stress-d3-z-p0.005-spatial_correlated | CSS routed | 7.9500% | 0.676 | 42 / 63 |
| stress-d3-z-p0.005-spatial_correlated | Temporal low-pass | 13.8450% | 1.484 | 400 / 1,600 |
| stress-d3-z-p0.005-spatial_correlated | Local peel + MWPM | 7.8450% | 0.676 | 0 / 0 |
| stress-d3-x-p0.005-spatial_correlated | Ordinary MWPM | 7.8000% | 0.509 | 0 / 0 |
| stress-d3-x-p0.005-spatial_correlated | Correlated MWPM | 8.0000% | 1.432 | 104 / 144 |
| stress-d3-x-p0.005-spatial_correlated | Union-find | 7.8200% | 1.358 | 7 / 11 |
| stress-d3-x-p0.005-spatial_correlated | Fast hybrid | 5.2500% | 0.030 | 733 / 223 |
| stress-d3-x-p0.005-spatial_correlated | CSS routed | 7.7500% | 0.588 | 40 / 30 |
| stress-d3-x-p0.005-spatial_correlated | Temporal low-pass | 15.3300% | 1.538 | 412 / 1,918 |
| stress-d3-x-p0.005-spatial_correlated | Local peel + MWPM | 7.8000% | 0.636 | 0 / 0 |
| stress-d5-z-p0.005-final_round_measurement | Ordinary MWPM | 1.3150% | 3.936 | 0 / 0 |
| stress-d5-z-p0.005-final_round_measurement | Correlated MWPM | 1.1350% | 10.835 | 106 / 70 |
| stress-d5-z-p0.005-final_round_measurement | Union-find | 1.4700% | 8.657 | 13 / 44 |
| stress-d5-z-p0.005-final_round_measurement | Fast hybrid | 1.3150% | 4.211 | 7 / 7 |
| stress-d5-z-p0.005-final_round_measurement | CSS routed | 1.2600% | 3.512 | 31 / 20 |
| stress-d5-z-p0.005-final_round_measurement | Temporal low-pass | 17.0550% | 7.583 | 100 / 3,248 |
| stress-d5-z-p0.005-final_round_measurement | Local peel + MWPM | 1.3100% | 4.534 | 1 / 0 |
| stress-d5-x-p0.005-final_round_measurement | Ordinary MWPM | 1.6200% | 3.725 | 0 / 0 |
| stress-d5-x-p0.005-final_round_measurement | Correlated MWPM | 1.2050% | 10.403 | 153 / 70 |
| stress-d5-x-p0.005-final_round_measurement | Union-find | 1.7350% | 8.310 | 25 / 48 |
| stress-d5-x-p0.005-final_round_measurement | Fast hybrid | 1.6250% | 4.070 | 12 / 13 |
| stress-d5-x-p0.005-final_round_measurement | CSS routed | 1.4850% | 3.631 | 51 / 24 |
| stress-d5-x-p0.005-final_round_measurement | Temporal low-pass | 17.8550% | 7.582 | 141 / 3,388 |
| stress-d5-x-p0.005-final_round_measurement | Local peel + MWPM | 1.6300% | 4.490 | 0 / 2 |
| stress-d5-z-p0.005-measurement_burst | Ordinary MWPM | 1.8650% | 4.163 | 0 / 0 |
| stress-d5-z-p0.005-measurement_burst | Correlated MWPM | 1.4850% | 11.997 | 160 / 84 |
| stress-d5-z-p0.005-measurement_burst | Union-find | 2.0200% | 9.000 | 33 / 64 |
| stress-d5-z-p0.005-measurement_burst | Fast hybrid | 1.8050% | 4.746 | 20 / 8 |
| stress-d5-z-p0.005-measurement_burst | CSS routed | 1.7500% | 3.642 | 41 / 18 |
| stress-d5-z-p0.005-measurement_burst | Temporal low-pass | 16.6350% | 8.706 | 150 / 3,104 |
| stress-d5-z-p0.005-measurement_burst | Local peel + MWPM | 1.8650% | 5.068 | 2 / 2 |
| stress-d5-x-p0.005-measurement_burst | Ordinary MWPM | 2.0600% | 4.621 | 0 / 0 |
| stress-d5-x-p0.005-measurement_burst | Correlated MWPM | 1.5000% | 11.597 | 189 / 77 |
| stress-d5-x-p0.005-measurement_burst | Union-find | 2.2200% | 9.441 | 48 / 80 |
| stress-d5-x-p0.005-measurement_burst | Fast hybrid | 2.0450% | 4.832 | 14 / 11 |
| stress-d5-x-p0.005-measurement_burst | CSS routed | 1.9050% | 4.080 | 50 / 19 |
| stress-d5-x-p0.005-measurement_burst | Temporal low-pass | 18.1200% | 8.742 | 152 / 3,364 |
| stress-d5-x-p0.005-measurement_burst | Local peel + MWPM | 2.0550% | 5.136 | 4 / 3 |
| stress-d5-z-p0.005-spatial_correlated | Ordinary MWPM | 6.2150% | 4.846 | 0 / 0 |
| stress-d5-z-p0.005-spatial_correlated | Correlated MWPM | 5.3800% | 13.205 | 345 / 178 |
| stress-d5-z-p0.005-spatial_correlated | Union-find | 6.2800% | 11.835 | 74 / 87 |
| stress-d5-z-p0.005-spatial_correlated | Fast hybrid | 6.1800% | 5.186 | 22 / 15 |
| stress-d5-z-p0.005-spatial_correlated | CSS routed | 6.1900% | 3.896 | 55 / 50 |
| stress-d5-z-p0.005-spatial_correlated | Temporal low-pass | 20.7450% | 7.697 | 464 / 3,370 |
| stress-d5-z-p0.005-spatial_correlated | Local peel + MWPM | 6.2150% | 5.480 | 0 / 0 |
| stress-d5-x-p0.005-spatial_correlated | Ordinary MWPM | 6.7350% | 5.096 | 0 / 0 |
| stress-d5-x-p0.005-spatial_correlated | Correlated MWPM | 5.9700% | 13.464 | 323 / 170 |
| stress-d5-x-p0.005-spatial_correlated | Union-find | 6.7950% | 11.931 | 89 / 101 |
| stress-d5-x-p0.005-spatial_correlated | Fast hybrid | 6.7000% | 5.493 | 18 / 11 |
| stress-d5-x-p0.005-spatial_correlated | CSS routed | 6.6750% | 4.450 | 46 / 34 |
| stress-d5-x-p0.005-spatial_correlated | Temporal low-pass | 22.1750% | 8.362 | 491 / 3,579 |
| stress-d5-x-p0.005-spatial_correlated | Local peel + MWPM | 6.7350% | 5.742 | 0 / 0 |
| stress-d7-z-p0.005-final_round_measurement | Ordinary MWPM | 0.9500% | 12.791 | 0 / 0 |
| stress-d7-z-p0.005-final_round_measurement | Correlated MWPM | 0.5950% | 34.860 | 117 / 46 |
| stress-d7-z-p0.005-final_round_measurement | Union-find | 1.0550% | 26.745 | 25 / 46 |
| stress-d7-z-p0.005-final_round_measurement | Fast hybrid | 0.9500% | 13.141 | 0 / 0 |
| stress-d7-z-p0.005-final_round_measurement | CSS routed | 0.8800% | 10.325 | 24 / 10 |
| stress-d7-z-p0.005-final_round_measurement | Temporal low-pass | 39.3050% | 31.498 | 97 / 7,768 |
| stress-d7-z-p0.005-final_round_measurement | Local peel + MWPM | 0.9550% | 16.609 | 1 / 2 |
| stress-d7-x-p0.005-final_round_measurement | Ordinary MWPM | 1.3450% | 12.815 | 0 / 0 |
| stress-d7-x-p0.005-final_round_measurement | Correlated MWPM | 0.6750% | 35.008 | 173 / 39 |
| stress-d7-x-p0.005-final_round_measurement | Union-find | 1.4950% | 27.252 | 31 / 61 |
| stress-d7-x-p0.005-final_round_measurement | Fast hybrid | 1.3450% | 12.929 | 0 / 0 |
| stress-d7-x-p0.005-final_round_measurement | CSS routed | 1.1800% | 10.495 | 42 / 9 |
| stress-d7-x-p0.005-final_round_measurement | Temporal low-pass | 40.9950% | 33.443 | 150 / 8,080 |
| stress-d7-x-p0.005-final_round_measurement | Local peel + MWPM | 1.3500% | 15.244 | 0 / 1 |
| stress-d7-z-p0.005-measurement_burst | Ordinary MWPM | 1.3250% | 14.506 | 0 / 0 |
| stress-d7-z-p0.005-measurement_burst | Correlated MWPM | 0.8000% | 38.861 | 144 / 39 |
| stress-d7-z-p0.005-measurement_burst | Union-find | 1.5200% | 30.126 | 49 / 88 |
| stress-d7-z-p0.005-measurement_burst | Fast hybrid | 1.3250% | 14.540 | 0 / 0 |
| stress-d7-z-p0.005-measurement_burst | CSS routed | 1.2050% | 10.704 | 30 / 6 |
| stress-d7-z-p0.005-measurement_burst | Temporal low-pass | 39.6400% | 32.679 | 132 / 7,795 |
| stress-d7-z-p0.005-measurement_burst | Local peel + MWPM | 1.3300% | 16.942 | 2 / 3 |
| stress-d7-x-p0.005-measurement_burst | Ordinary MWPM | 1.5850% | 14.149 | 0 / 0 |
| stress-d7-x-p0.005-measurement_burst | Correlated MWPM | 0.9250% | 37.613 | 181 / 49 |
| stress-d7-x-p0.005-measurement_burst | Union-find | 1.8850% | 29.845 | 38 / 98 |
| stress-d7-x-p0.005-measurement_burst | Fast hybrid | 1.5850% | 14.545 | 0 / 0 |
| stress-d7-x-p0.005-measurement_burst | CSS routed | 1.4400% | 12.286 | 38 / 9 |
| stress-d7-x-p0.005-measurement_burst | Temporal low-pass | 40.8450% | 34.873 | 165 / 8,017 |
| stress-d7-x-p0.005-measurement_burst | Local peel + MWPM | 1.6000% | 16.572 | 1 / 4 |
| stress-d7-z-p0.005-spatial_correlated | Ordinary MWPM | 6.2300% | 17.788 | 0 / 0 |
| stress-d7-z-p0.005-spatial_correlated | Correlated MWPM | 5.0600% | 46.450 | 445 / 211 |
| stress-d7-z-p0.005-spatial_correlated | Union-find | 6.5500% | 41.840 | 125 / 189 |
| stress-d7-z-p0.005-spatial_correlated | Fast hybrid | 6.2300% | 18.393 | 0 / 0 |
| stress-d7-z-p0.005-spatial_correlated | CSS routed | 6.0550% | 13.597 | 57 / 22 |
| stress-d7-z-p0.005-spatial_correlated | Temporal low-pass | 41.7450% | 31.567 | 632 / 7,735 |
| stress-d7-z-p0.005-spatial_correlated | Local peel + MWPM | 6.2250% | 20.473 | 2 / 1 |
| stress-d7-x-p0.005-spatial_correlated | Ordinary MWPM | 7.4950% | 18.824 | 0 / 0 |
| stress-d7-x-p0.005-spatial_correlated | Correlated MWPM | 5.7650% | 47.630 | 607 / 261 |
| stress-d7-x-p0.005-spatial_correlated | Union-find | 7.6700% | 42.154 | 191 / 226 |
| stress-d7-x-p0.005-spatial_correlated | Fast hybrid | 7.4950% | 18.581 | 0 / 0 |
| stress-d7-x-p0.005-spatial_correlated | CSS routed | 7.3300% | 15.625 | 91 / 58 |
| stress-d7-x-p0.005-spatial_correlated | Temporal low-pass | 43.5200% | 32.301 | 712 / 7,917 |
| stress-d7-x-p0.005-spatial_correlated | Local peel + MWPM | 7.4950% | 21.247 | 0 / 0 |

## Finite-distance crossings

The sweep yielded 24 point-estimate reversals and 23 Wilson-supported brackets for adjacent distances. See [crossings.csv](crossings.csv) for physical-noise intervals.

Point crossings interpolate the difference in per-memory logical error between sampled noise probabilities on the log-p axis. A Wilson-supported bracket requires disjoint 95% Wilson intervals with opposite order at its two ends. The bracket is descriptive and the study measures only finite distances with rounds equal to distance.
