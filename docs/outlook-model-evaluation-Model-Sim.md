# Outlook model evaluation: Model-Sim (Standard27)

Generated 2026-11-02 by `scripts/train_outlook_models.py` from seasons 2045-2086 of the long CPU league (trained in 24 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least five seasons earlier (for horizon 1 the training seasons end 2069, the test seasons are 2080-2085; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 1 | 124791 | 0.981 | 0.961 | 0.974 | 2.93 | 4.02 | 3.43 | +6% / +15% / +20% | 77% | 81% | 48% |
| hitters | 2 | 104330 | 0.954 | 0.875 | 0.930 | 4.62 | 7.45 | 5.70 | +9% / +19% / +24% | 76% | 80% | 47% |
| hitters | 3 | 83913 | 0.918 | 0.744 | 0.874 | 6.14 | 10.93 | 7.59 | +10% / +19% / +23% | 76% | 80% | 47% |
| hitters | 5 | 50063 | 0.813 | 0.322 | 0.731 | 8.70 | 17.87 | 10.82 | +14% / +20% / +14% | 73% | 80% | 45% |
| hitters | 7 | 27762 | 0.654 | -0.715 | 0.536 | 10.01 | 24.13 | 12.20 | +18% / +18% / +6% | 69% | 79% | 41% |
| hitters | 10 | 12767 | 0.462 | -3.115 | 0.166 | 9.60 | 28.10 | 12.47 | +25% / +23% / +11% | 64% | 81% | 36% |
| pitchers | 1 | 47445 | 0.975 | 0.954 | 0.964 | 4.19 | 5.33 | 4.90 | +6% / +14% / +36% | 76% | 80% | 46% |
| pitchers | 2 | 39136 | 0.957 | 0.881 | 0.923 | 5.97 | 9.49 | 7.83 | +8% / +24% / +37% | 78% | 80% | 49% |
| pitchers | 3 | 30743 | 0.933 | 0.768 | 0.868 | 7.74 | 14.22 | 10.81 | +11% / +28% / +38% | 77% | 79% | 48% |
| pitchers | 5 | 18222 | 0.867 | 0.377 | 0.736 | 10.49 | 24.36 | 15.78 | +18% / +34% / +29% | 76% | 81% | 47% |
| pitchers | 7 | 10148 | 0.727 | -0.779 | 0.562 | 11.76 | 33.50 | 16.79 | +23% / +30% / +20% | 72% | 81% | 44% |
| pitchers | 10 | 4758 | 0.468 | -6.811 | -0.111 | 8.38 | 35.28 | 13.28 | +29% / +37% / +32% | 66% | 80% | 39% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|
| hitters | 16-19 | 37450 | 6.15 | 12.20 | 76% |
| hitters | 20-22 | 21543 | 7.77 | 13.95 | 77% |
| hitters | 23-26 | 14761 | 4.94 | 5.87 | 75% |
| hitters | 27-30 | 6802 | 4.28 | 5.90 | 74% |
| hitters | 31+ | 3338 | 4.69 | 9.80 | 77% |
| pitchers | 16-19 | 14692 | 7.40 | 11.56 | 76% |
| pitchers | 20-22 | 7222 | 11.04 | 26.63 | 79% |
| pitchers | 23-26 | 4713 | 7.43 | 12.50 | 78% |
| pitchers | 27-30 | 2096 | 3.54 | 3.74 | 75% |
| pitchers | 31+ | 2009 | 3.45 | 4.12 | 73% |

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 153719 | 1.749% | 1.695% | 0.998 | 0.970 | +70% | +69% |
| hitters | 1 | top10 | 153719 | 0.351% | 0.299% | 0.999 | 0.962 | +67% | +74% |
| hitters | 2 | top50 | 154025 | 1.723% | 1.668% | 0.995 | 0.956 | +56% | +58% |
| hitters | 2 | top10 | 154025 | 0.348% | 0.293% | 0.997 | 0.951 | +53% | +62% |
| hitters | 3 | top50 | 154275 | 1.646% | 1.598% | 0.990 | 0.933 | +45% | +50% |
| hitters | 3 | top10 | 154275 | 0.329% | 0.294% | 0.994 | 0.931 | +40% | +55% |
| hitters | 5 | top50 | 154826 | 1.405% | 1.327% | 0.976 | 0.872 | +30% | +40% |
| hitters | 5 | top10 | 154826 | 0.284% | 0.245% | 0.982 | 0.882 | +25% | +41% |
| hitters | 7 | top50 | 155168 | 1.049% | 1.069% | 0.967 | 0.831 | +21% | +35% |
| hitters | 7 | top10 | 155168 | 0.214% | 0.190% | 0.974 | 0.850 | +13% | +34% |
| hitters | 10 | top50 | 155077 | 0.547% | 0.576% | 0.954 | 0.782 | +9% | +29% |
| hitters | 10 | top10 | 155077 | 0.098% | 0.077% | 0.951 | 0.805 | +1% | +20% |
| pitchers | 1 | top50 | 59298 | 0.993% | 0.947% | 0.998 | 0.957 | +65% | +67% |
| pitchers | 1 | top10 | 59298 | 0.199% | 0.192% | 0.999 | 0.953 | +56% | +69% |
| pitchers | 2 | top50 | 59443 | 0.977% | 0.933% | 0.996 | 0.955 | +53% | +56% |
| pitchers | 2 | top10 | 59443 | 0.195% | 0.189% | 0.998 | 0.952 | +43% | +59% |
| pitchers | 3 | top50 | 59588 | 0.952% | 0.904% | 0.993 | 0.950 | +44% | +49% |
| pitchers | 3 | top10 | 59588 | 0.190% | 0.166% | 0.996 | 0.948 | +34% | +52% |
| pitchers | 5 | top50 | 59637 | 0.882% | 0.841% | 0.986 | 0.927 | +30% | +40% |
| pitchers | 5 | top10 | 59637 | 0.174% | 0.135% | 0.992 | 0.930 | +20% | +40% |
| pitchers | 7 | top50 | 59482 | 0.782% | 0.736% | 0.969 | 0.888 | +21% | +33% |
| pitchers | 7 | top10 | 59482 | 0.156% | 0.123% | 0.967 | 0.876 | +11% | +30% |
| pitchers | 10 | top50 | 59419 | 0.560% | 0.514% | 0.950 | 0.837 | +13% | +28% |
| pitchers | 10 | top10 | 59419 | 0.104% | 0.071% | 0.917 | 0.754 | +5% | +25% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 1.756% | 1.692% | 1.741% | 0.00396 | 0.00396 |
| hitters | 1 | top10 | 0.352% | 0.301% | 0.347% | 0.00110 | 0.00105 |
| hitters | 2 | top50 | 1.737% | 1.678% | 1.731% | 0.00606 | 0.00607 |
| hitters | 2 | top10 | 0.349% | 0.287% | 0.336% | 0.00160 | 0.00159 |
| hitters | 3 | top50 | 1.669% | 1.625% | 1.667% | 0.00774 | 0.00775 |
| hitters | 3 | top10 | 0.331% | 0.281% | 0.304% | 0.00191 | 0.00191 |
| hitters | 5 | top50 | 1.402% | 1.339% | 1.416% | 0.00883 | 0.00879 |
| hitters | 5 | top10 | 0.266% | 0.249% | 0.304% | 0.00188 | 0.00187 |
| hitters | 7 | top50 | 1.070% | 1.059% | 1.027% | 0.00824 | 0.00824 |
| hitters | 7 | top10 | 0.216% | 0.189% | 0.209% | 0.00182 | 0.00187 |
| hitters | 10 | top50 | 0.528% | 0.597% | 0.597% | 0.00475 | 0.00481 |
| hitters | 10 | top10 | 0.112% | 0.080% | 0.084% | 0.00109 | 0.00109 |
| pitchers | 1 | top50 | 0.996% | 0.943% | 0.972% | 0.00345 | 0.00360 |
| pitchers | 1 | top10 | 0.199% | 0.184% | 0.186% | 0.00091 | 0.00093 |
| pitchers | 2 | top50 | 0.987% | 0.922% | 0.951% | 0.00438 | 0.00443 |
| pitchers | 2 | top10 | 0.199% | 0.189% | 0.185% | 0.00118 | 0.00121 |
| pitchers | 3 | top50 | 0.957% | 0.890% | 0.923% | 0.00492 | 0.00502 |
| pitchers | 3 | top10 | 0.195% | 0.164% | 0.177% | 0.00129 | 0.00134 |
| pitchers | 5 | top50 | 0.876% | 0.842% | 0.903% | 0.00553 | 0.00543 |
| pitchers | 5 | top10 | 0.171% | 0.134% | 0.168% | 0.00139 | 0.00140 |
| pitchers | 7 | top50 | 0.770% | 0.734% | 0.790% | 0.00562 | 0.00571 |
| pitchers | 7 | top10 | 0.150% | 0.125% | 0.167% | 0.00134 | 0.00139 |
| pitchers | 10 | top50 | 0.582% | 0.507% | 0.527% | 0.00512 | 0.00526 |
| pitchers | 10 | top10 | 0.129% | 0.068% | 0.074% | 0.00124 | 0.00128 |

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
| 7 | 0.0005 | 0.0001 |
| 8 | 0.0010 | 0.0003 |
| 9 | 0.0031 | 0.0023 |
| 10 | 0.1547 | 0.1620 |

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
| 8 | 0.0002 | 0.0002 |
| 9 | 0.0009 | 0.0003 |
| 10 | 0.0891 | 0.0947 |

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 62027 | 80% | 0.840 | +22% | 48095 | 0.203 | 0.244 | 0.244 | 78% |
| 2 | 62125 | 66% | 0.844 | +28% | 40424 | 0.205 | 0.243 | 0.243 | 79% |
| 3 | 62235 | 52% | 0.848 | +31% | 32356 | 0.206 | 0.241 | 0.240 | 78% |
| 5 | 62360 | 31% | 0.858 | +32% | 19149 | 0.212 | 0.242 | 0.241 | 79% |
| 7 | 62279 | 17% | 0.877 | +31% | 10588 | 0.208 | 0.234 | 0.232 | 78% |
| 10 | 62196 | 8% | 0.904 | +27% | 4912 | 0.212 | 0.232 | 0.229 | 78% |

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
| 1 | 0.626 | 0.602 | 0.575 |
| 2 | 0.631 | 0.607 | 0.574 |
| 3 | 0.642 | 0.617 | 0.577 |
| 5 | 0.669 | 0.640 | 0.579 |
| 7 | 0.703 | 0.653 | 0.587 |
| 10 | 0.717 | 0.658 | 0.592 |
