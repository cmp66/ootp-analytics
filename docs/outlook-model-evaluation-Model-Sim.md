# Outlook model evaluation: Model-Sim (Standard27)

Generated 2026-11-02 by `scripts/train_outlook_models.py` from seasons 2045-2086 of the long CPU league (trained in 26 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least five seasons earlier (for horizon 1 the training seasons end 2074, the test seasons are 2080-2085; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 1 | 124791 | 0.981 | 0.961 | 0.974 | 2.92 | 4.02 | 3.43 | +6% / +15% / +20% | 77% | 81% | 48% |
| hitters | 2 | 104330 | 0.954 | 0.875 | 0.930 | 4.62 | 7.45 | 5.69 | +9% / +19% / +24% | 76% | 80% | 47% |
| hitters | 3 | 83913 | 0.918 | 0.744 | 0.874 | 6.12 | 10.93 | 7.59 | +11% / +19% / +23% | 76% | 80% | 47% |
| hitters | 5 | 50063 | 0.813 | 0.322 | 0.731 | 8.69 | 17.87 | 10.82 | +15% / +20% / +15% | 74% | 79% | 46% |
| pitchers | 1 | 47445 | 0.975 | 0.954 | 0.964 | 4.18 | 5.33 | 4.90 | +6% / +15% / +36% | 77% | 80% | 48% |
| pitchers | 2 | 39136 | 0.957 | 0.881 | 0.923 | 5.97 | 9.49 | 7.83 | +8% / +24% / +37% | 79% | 80% | 50% |
| pitchers | 3 | 30743 | 0.933 | 0.768 | 0.868 | 7.72 | 14.22 | 10.81 | +11% / +29% / +38% | 78% | 79% | 48% |
| pitchers | 5 | 18222 | 0.867 | 0.377 | 0.736 | 10.48 | 24.36 | 15.78 | +18% / +34% / +29% | 76% | 80% | 48% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|
| hitters | 16-19 | 37450 | 6.15 | 12.20 | 76% |
| hitters | 20-22 | 21543 | 7.72 | 13.95 | 77% |
| hitters | 23-26 | 14761 | 4.88 | 5.87 | 76% |
| hitters | 27-30 | 6802 | 4.28 | 5.90 | 75% |
| hitters | 31+ | 3338 | 4.69 | 9.80 | 79% |
| pitchers | 16-19 | 14692 | 7.37 | 11.56 | 77% |
| pitchers | 20-22 | 7222 | 11.04 | 26.63 | 79% |
| pitchers | 23-26 | 4713 | 7.41 | 12.50 | 80% |
| pitchers | 27-30 | 2096 | 3.52 | 3.74 | 77% |
| pitchers | 31+ | 2009 | 3.44 | 4.12 | 75% |

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 153719 | 1.749% | 1.711% | 0.998 | 0.970 | +70% | +69% |
| hitters | 1 | top10 | 153719 | 0.351% | 0.313% | 0.999 | 0.962 | +67% | +74% |
| hitters | 2 | top50 | 154025 | 1.723% | 1.678% | 0.995 | 0.956 | +56% | +58% |
| hitters | 2 | top10 | 154025 | 0.348% | 0.286% | 0.997 | 0.950 | +54% | +63% |
| hitters | 3 | top50 | 154275 | 1.646% | 1.572% | 0.990 | 0.933 | +45% | +51% |
| hitters | 3 | top10 | 154275 | 0.329% | 0.259% | 0.994 | 0.931 | +41% | +55% |
| hitters | 5 | top50 | 154826 | 1.405% | 1.328% | 0.975 | 0.872 | +31% | +40% |
| hitters | 5 | top10 | 154826 | 0.284% | 0.223% | 0.986 | 0.882 | +22% | +40% |
| pitchers | 1 | top50 | 59298 | 0.993% | 0.951% | 0.998 | 0.957 | +65% | +67% |
| pitchers | 1 | top10 | 59298 | 0.199% | 0.201% | 0.999 | 0.953 | +60% | +71% |
| pitchers | 2 | top50 | 59443 | 0.977% | 0.927% | 0.996 | 0.954 | +53% | +56% |
| pitchers | 2 | top10 | 59443 | 0.195% | 0.188% | 0.998 | 0.952 | +47% | +61% |
| pitchers | 3 | top50 | 59588 | 0.952% | 0.909% | 0.993 | 0.950 | +44% | +50% |
| pitchers | 3 | top10 | 59588 | 0.190% | 0.178% | 0.997 | 0.948 | +38% | +54% |
| pitchers | 5 | top50 | 59637 | 0.882% | 0.831% | 0.987 | 0.927 | +31% | +41% |
| pitchers | 5 | top10 | 59637 | 0.174% | 0.144% | 0.993 | 0.930 | +24% | +43% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 1.756% | 1.704% | 1.735% | 0.00402 | 0.00401 |
| hitters | 1 | top10 | 0.352% | 0.308% | 0.341% | 0.00116 | 0.00115 |
| hitters | 2 | top50 | 1.737% | 1.687% | 1.735% | 0.00603 | 0.00606 |
| hitters | 2 | top10 | 0.349% | 0.287% | 0.351% | 0.00159 | 0.00160 |
| hitters | 3 | top50 | 1.669% | 1.597% | 1.658% | 0.00769 | 0.00768 |
| hitters | 3 | top10 | 0.331% | 0.248% | 0.304% | 0.00190 | 0.00187 |
| hitters | 5 | top50 | 1.402% | 1.338% | 1.414% | 0.00878 | 0.00878 |
| hitters | 5 | top10 | 0.266% | 0.229% | 0.311% | 0.00193 | 0.00194 |
| pitchers | 1 | top50 | 0.996% | 0.946% | 0.979% | 0.00344 | 0.00349 |
| pitchers | 1 | top10 | 0.199% | 0.194% | 0.181% | 0.00084 | 0.00084 |
| pitchers | 2 | top50 | 0.987% | 0.920% | 0.952% | 0.00438 | 0.00444 |
| pitchers | 2 | top10 | 0.199% | 0.186% | 0.181% | 0.00106 | 0.00116 |
| pitchers | 3 | top50 | 0.957% | 0.899% | 0.925% | 0.00490 | 0.00492 |
| pitchers | 3 | top10 | 0.195% | 0.181% | 0.188% | 0.00125 | 0.00131 |
| pitchers | 5 | top50 | 0.876% | 0.828% | 0.892% | 0.00547 | 0.00542 |
| pitchers | 5 | top10 | 0.171% | 0.144% | 0.181% | 0.00134 | 0.00142 |

Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):

hitters:

| Group | Predicted | Observed |
|---:|---:|---:|
| 1 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 |
| 3 | 0.0001 | 0.0000 |
| 4 | 0.0001 | 0.0000 |
| 5 | 0.0001 | 0.0000 |
| 6 | 0.0003 | 0.0000 |
| 7 | 0.0005 | 0.0000 |
| 8 | 0.0010 | 0.0002 |
| 9 | 0.0031 | 0.0022 |
| 10 | 0.1521 | 0.1622 |

pitchers:

| Group | Predicted | Observed |
|---:|---:|---:|
| 1 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 0.0000 |
| 7 | 0.0001 | 0.0000 |
| 8 | 0.0002 | 0.0000 |
| 9 | 0.0009 | 0.0002 |
| 10 | 0.0897 | 0.0950 |

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62027 | 80% | 0.841 | +22% | 48095 | 0.203 | 0.244 | 0.244 | 79% |
| 2 | 62125 | 66% | 0.844 | +28% | 40424 | 0.204 | 0.243 | 0.243 | 79% |
| 3 | 62235 | 52% | 0.849 | +31% | 32356 | 0.205 | 0.241 | 0.240 | 79% |
| 5 | 62360 | 31% | 0.859 | +32% | 19149 | 0.211 | 0.242 | 0.241 | 78% |

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
| 1 | 0.629 | 0.601 | 0.574 |
| 2 | 0.636 | 0.605 | 0.571 |
| 3 | 0.645 | 0.616 | 0.575 |
| 5 | 0.676 | 0.637 | 0.579 |
