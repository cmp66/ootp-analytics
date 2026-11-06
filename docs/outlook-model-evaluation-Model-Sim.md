# Outlook model evaluation: Model-Sim (Standard27)

Generated 2026-11-06 by `scripts/train_outlook_models.py` from seasons 2045-2086 of the long CPU league (trained in 16 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least ten seasons earlier (for horizon 1 the training seasons end 2069, the test seasons are 2080-2085; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 1 | 124791 | 0.981 | 0.961 | 0.974 | 2.91 | 4.02 | 3.43 | +7% / +15% / +21% | 77% | 81% | 47% |
| hitters | 5 | 50063 | 0.816 | 0.322 | 0.731 | 8.64 | 17.87 | 10.82 | +15% / +20% / +15% | 73% | 80% | 45% |
| hitters | 7 | 27762 | 0.658 | -0.715 | 0.536 | 9.97 | 24.13 | 12.20 | +18% / +18% / +7% | 69% | 79% | 41% |
| hitters | 10 | 12767 | 0.467 | -3.115 | 0.166 | 9.50 | 28.10 | 12.47 | +25% / +24% / +10% | 64% | 80% | 37% |
| hitters | 99 | 131303 | 0.874 | 0.644 | 0.796 | 6.66 | 11.21 | 8.38 | +8% / +21% / +27% | 76% | 79% | 47% |
| pitchers | 1 | 47445 | 0.975 | 0.954 | 0.964 | 4.18 | 5.33 | 4.90 | +6% / +15% / +37% | 76% | 80% | 46% |
| pitchers | 5 | 18222 | 0.868 | 0.377 | 0.736 | 10.47 | 24.36 | 15.78 | +18% / +34% / +29% | 76% | 81% | 47% |
| pitchers | 7 | 10148 | 0.727 | -0.779 | 0.562 | 11.76 | 33.50 | 16.79 | +23% / +30% / +19% | 72% | 81% | 45% |
| pitchers | 10 | 4758 | 0.464 | -6.811 | -0.111 | 8.45 | 35.28 | 13.28 | +30% / +36% / +32% | 66% | 79% | 40% |
| pitchers | 99 | 49569 | 0.867 | 0.639 | 0.728 | 9.87 | 15.93 | 14.16 | +8% / +30% / +38% | 73% | 82% | 43% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 153719 | 1.749% | 1.693% | 0.998 | 0.970 | +70% | +70% |
| hitters | 1 | top10 | 153719 | 0.351% | 0.303% | 0.999 | 0.962 | +68% | +74% |
| hitters | 5 | top50 | 154826 | 1.405% | 1.311% | 0.978 | 0.872 | +31% | +42% |
| hitters | 5 | top10 | 154826 | 0.284% | 0.241% | 0.982 | 0.882 | +26% | +41% |
| hitters | 7 | top50 | 155168 | 1.049% | 1.060% | 0.971 | 0.831 | +21% | +37% |
| hitters | 7 | top10 | 155168 | 0.214% | 0.188% | 0.977 | 0.850 | +14% | +37% |
| hitters | 10 | top50 | 155077 | 0.547% | 0.569% | 0.956 | 0.782 | +10% | +30% |
| hitters | 10 | top10 | 155077 | 0.098% | 0.076% | 0.957 | 0.805 | +2% | +22% |
| hitters | 99 | top50 | 155077 | 2.733% | 2.658% | 0.980 | 0.901 | +47% | +49% |
| hitters | 99 | top10 | 155077 | 0.602% | 0.528% | 0.985 | 0.894 | +45% | +53% |
| pitchers | 1 | top50 | 59298 | 0.993% | 0.948% | 0.998 | 0.957 | +65% | +67% |
| pitchers | 1 | top10 | 59298 | 0.199% | 0.192% | 0.999 | 0.953 | +56% | +69% |
| pitchers | 5 | top50 | 59637 | 0.882% | 0.842% | 0.986 | 0.927 | +29% | +40% |
| pitchers | 5 | top10 | 59637 | 0.174% | 0.139% | 0.991 | 0.930 | +21% | +40% |
| pitchers | 7 | top50 | 59482 | 0.782% | 0.734% | 0.969 | 0.888 | +21% | +33% |
| pitchers | 7 | top10 | 59482 | 0.156% | 0.122% | 0.972 | 0.876 | +11% | +31% |
| pitchers | 10 | top50 | 59419 | 0.560% | 0.511% | 0.951 | 0.837 | +13% | +28% |
| pitchers | 10 | top10 | 59419 | 0.104% | 0.070% | 0.922 | 0.754 | +6% | +26% |
| pitchers | 99 | top50 | 59419 | 1.967% | 1.933% | 0.973 | 0.901 | +42% | +42% |
| pitchers | 99 | top10 | 59419 | 0.402% | 0.403% | 0.964 | 0.872 | +29% | +40% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 1.756% | 1.690% | 1.740% | 0.00394 | 0.00394 |
| hitters | 1 | top10 | 0.352% | 0.305% | 0.347% | 0.00108 | 0.00104 |
| hitters | 5 | top50 | 1.402% | 1.319% | 1.413% | 0.00869 | 0.00867 |
| hitters | 5 | top10 | 0.266% | 0.243% | 0.299% | 0.00186 | 0.00186 |
| hitters | 7 | top50 | 1.070% | 1.043% | 1.010% | 0.00817 | 0.00815 |
| hitters | 7 | top10 | 0.216% | 0.194% | 0.222% | 0.00179 | 0.00184 |
| hitters | 10 | top50 | 0.528% | 0.590% | 0.607% | 0.00470 | 0.00478 |
| hitters | 10 | top10 | 0.112% | 0.079% | 0.085% | 0.00108 | 0.00109 |
| hitters | 99 | top50 | 2.768% | 2.702% | 2.791% | 0.01148 | 0.01148 |
| hitters | 99 | top10 | 0.622% | 0.511% | 0.548% | 0.00332 | 0.00331 |
| pitchers | 1 | top50 | 0.996% | 0.948% | 0.984% | 0.00344 | 0.00353 |
| pitchers | 1 | top10 | 0.199% | 0.183% | 0.185% | 0.00093 | 0.00093 |
| pitchers | 5 | top50 | 0.876% | 0.847% | 0.899% | 0.00553 | 0.00544 |
| pitchers | 5 | top10 | 0.171% | 0.137% | 0.168% | 0.00138 | 0.00140 |
| pitchers | 7 | top50 | 0.770% | 0.729% | 0.778% | 0.00561 | 0.00569 |
| pitchers | 7 | top10 | 0.150% | 0.124% | 0.165% | 0.00133 | 0.00140 |
| pitchers | 10 | top50 | 0.582% | 0.503% | 0.525% | 0.00510 | 0.00517 |
| pitchers | 10 | top10 | 0.129% | 0.068% | 0.073% | 0.00121 | 0.00126 |
| pitchers | 99 | top50 | 1.973% | 1.912% | 1.931% | 0.01019 | 0.01020 |
| pitchers | 99 | top10 | 0.426% | 0.402% | 0.375% | 0.00318 | 0.00318 |

Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62027 | 80% | 0.845 | +23% | 48095 | 0.202 | 0.244 | 0.244 | 79% |
| 5 | 62360 | 31% | 0.863 | +33% | 19149 | 0.211 | 0.242 | 0.241 | 79% |
| 7 | 62279 | 17% | 0.881 | +32% | 10588 | 0.209 | 0.234 | 0.232 | 78% |
| 10 | 62196 | 8% | 0.905 | +28% | 4912 | 0.211 | 0.232 | 0.229 | 78% |

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
| 1 | 0.624 | 0.602 | 0.574 |
| 5 | 0.669 | 0.640 | 0.579 |
| 7 | 0.703 | 0.651 | 0.588 |
| 10 | 0.721 | 0.655 | 0.593 |
