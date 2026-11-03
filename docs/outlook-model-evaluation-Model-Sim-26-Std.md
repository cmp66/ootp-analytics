# Outlook model evaluation: Model-Sim-26-Std (Standard26)

Generated 2026-11-02 by `scripts/train_outlook_models.py` from seasons 2037-2099 of the long CPU league (trained in 74 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least five seasons earlier (for horizon 1 the training seasons end 2082, the test seasons are 2093-2098; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 1 | 98125 | 0.967 | 0.906 | 0.961 | 3.82 | 6.31 | 4.19 | +9% / +9% / +8% | 78% | 80% | 49% |
| hitters | 2 | 85909 | 0.916 | 0.647 | 0.896 | 5.58 | 11.63 | 6.30 | +13% / +11% / +10% | 78% | 81% | 49% |
| hitters | 3 | 74455 | 0.853 | 0.187 | 0.815 | 6.75 | 16.48 | 7.70 | +15% / +12% / +12% | 78% | 81% | 49% |
| hitters | 5 | 53263 | 0.734 | -1.222 | 0.639 | 7.96 | 24.25 | 9.54 | +17% / +17% / +19% | 77% | 81% | 47% |
| hitters | 7 | 36275 | 0.635 | -3.091 | 0.434 | 8.24 | 29.32 | 10.63 | +25% / +22% / +23% | 75% | 81% | 45% |
| hitters | 10 | 19508 | 0.627 | -4.367 | 0.263 | 8.31 | 33.91 | 12.17 | +34% / +32% / +30% | 70% | 81% | 42% |
| pitchers | 1 | 26040 | 0.949 | 0.919 | 0.942 | 2.81 | 3.42 | 3.00 | +5% / +6% / +12% | 78% | 79% | 49% |
| pitchers | 2 | 23005 | 0.904 | 0.784 | 0.889 | 3.82 | 5.68 | 4.16 | +6% / +8% / +11% | 78% | 80% | 49% |
| pitchers | 3 | 20092 | 0.852 | 0.561 | 0.823 | 4.55 | 7.99 | 5.02 | +7% / +9% / +12% | 78% | 79% | 48% |
| pitchers | 5 | 14660 | 0.745 | -0.113 | 0.687 | 5.51 | 12.20 | 6.20 | +10% / +11% / +13% | 76% | 79% | 47% |
| pitchers | 7 | 10045 | 0.631 | -1.166 | 0.535 | 5.96 | 15.43 | 6.83 | +11% / +13% / +15% | 74% | 79% | 45% |
| pitchers | 10 | 5013 | 0.471 | -3.121 | 0.161 | 5.66 | 16.47 | 7.40 | +21% / +24% / +19% | 70% | 79% | 43% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|
| hitters | 16-19 | 16080 | 9.17 | 31.51 | 79% |
| hitters | 20-22 | 21152 | 8.51 | 21.38 | 77% |
| hitters | 23-26 | 21879 | 4.79 | 6.22 | 78% |
| hitters | 27-30 | 11205 | 4.39 | 6.75 | 80% |
| hitters | 31+ | 4139 | 5.07 | 13.56 | 78% |
| pitchers | 16-19 | 3064 | 5.50 | 12.22 | 78% |
| pitchers | 20-22 | 6002 | 5.04 | 10.60 | 79% |
| pitchers | 23-26 | 6322 | 4.53 | 6.15 | 77% |
| pitchers | 27-30 | 2571 | 3.18 | 3.53 | 78% |
| pitchers | 31+ | 2133 | 3.46 | 5.38 | 77% |

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 114759 | 2.477% | 2.400% | 0.997 | 0.969 | +68% | +65% |
| hitters | 1 | top10 | 114759 | 0.514% | 0.463% | 0.999 | 0.963 | +67% | +72% |
| hitters | 2 | top50 | 114391 | 2.446% | 2.330% | 0.994 | 0.949 | +57% | +58% |
| hitters | 2 | top10 | 114391 | 0.511% | 0.417% | 0.998 | 0.956 | +53% | +62% |
| hitters | 3 | top50 | 114184 | 2.378% | 2.228% | 0.990 | 0.916 | +49% | +54% |
| hitters | 3 | top10 | 114184 | 0.501% | 0.382% | 0.997 | 0.930 | +42% | +58% |
| hitters | 5 | top50 | 113821 | 2.106% | 1.917% | 0.985 | 0.838 | +42% | +52% |
| hitters | 5 | top10 | 113821 | 0.448% | 0.312% | 0.994 | 0.850 | +31% | +54% |
| hitters | 7 | top50 | 113684 | 1.688% | 1.547% | 0.983 | 0.795 | +36% | +52% |
| hitters | 7 | top10 | 113684 | 0.341% | 0.237% | 0.992 | 0.813 | +23% | +51% |
| hitters | 10 | top50 | 115083 | 0.964% | 0.851% | 0.986 | 0.725 | +31% | +54% |
| hitters | 10 | top10 | 115083 | 0.189% | 0.128% | 0.994 | 0.700 | +18% | +53% |
| pitchers | 1 | top50 | 30214 | 2.079% | 2.027% | 0.997 | 0.963 | +67% | +65% |
| pitchers | 1 | top10 | 30214 | 0.424% | 0.384% | 0.999 | 0.956 | +66% | +74% |
| pitchers | 2 | top50 | 30210 | 2.046% | 1.925% | 0.995 | 0.958 | +55% | +55% |
| pitchers | 2 | top10 | 30210 | 0.414% | 0.350% | 0.998 | 0.956 | +58% | +67% |
| pitchers | 3 | top50 | 30261 | 1.993% | 1.840% | 0.993 | 0.943 | +49% | +53% |
| pitchers | 3 | top10 | 30261 | 0.397% | 0.326% | 0.998 | 0.951 | +47% | +62% |
| pitchers | 5 | top50 | 29985 | 1.784% | 1.620% | 0.988 | 0.898 | +41% | +52% |
| pitchers | 5 | top10 | 29985 | 0.347% | 0.267% | 0.996 | 0.907 | +35% | +56% |
| pitchers | 7 | top50 | 29735 | 1.423% | 1.214% | 0.987 | 0.840 | +37% | +52% |
| pitchers | 7 | top10 | 29735 | 0.276% | 0.177% | 0.996 | 0.869 | +35% | +60% |
| pitchers | 10 | top50 | 30183 | 0.795% | 0.674% | 0.989 | 0.783 | +32% | +54% |
| pitchers | 10 | top10 | 30183 | 0.192% | 0.093% | 0.996 | 0.806 | +23% | +55% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 2.475% | 2.379% | 2.439% | 0.00595 | 0.00597 |
| hitters | 1 | top10 | 0.512% | 0.450% | 0.487% | 0.00168 | 0.00169 |
| hitters | 2 | top50 | 2.425% | 2.265% | 2.351% | 0.00809 | 0.00806 |
| hitters | 2 | top10 | 0.501% | 0.395% | 0.476% | 0.00246 | 0.00242 |
| hitters | 3 | top50 | 2.351% | 2.195% | 2.327% | 0.01007 | 0.01004 |
| hitters | 3 | top10 | 0.497% | 0.378% | 0.504% | 0.00287 | 0.00276 |
| hitters | 5 | top50 | 2.066% | 1.899% | 2.091% | 0.01113 | 0.01111 |
| hitters | 5 | top10 | 0.469% | 0.326% | 0.459% | 0.00321 | 0.00317 |
| hitters | 7 | top50 | 1.658% | 1.505% | 1.626% | 0.01063 | 0.01063 |
| hitters | 7 | top10 | 0.337% | 0.236% | 0.343% | 0.00265 | 0.00267 |
| hitters | 10 | top50 | 0.895% | 0.773% | 0.874% | 0.00636 | 0.00635 |
| hitters | 10 | top10 | 0.154% | 0.098% | 0.153% | 0.00135 | 0.00137 |
| pitchers | 1 | top50 | 2.131% | 2.031% | 2.035% | 0.00568 | 0.00578 |
| pitchers | 1 | top10 | 0.440% | 0.395% | 0.436% | 0.00143 | 0.00147 |
| pitchers | 2 | top50 | 2.074% | 1.920% | 2.007% | 0.00747 | 0.00749 |
| pitchers | 2 | top10 | 0.428% | 0.348% | 0.384% | 0.00165 | 0.00180 |
| pitchers | 3 | top50 | 2.004% | 1.818% | 1.910% | 0.00841 | 0.00862 |
| pitchers | 3 | top10 | 0.417% | 0.319% | 0.358% | 0.00216 | 0.00220 |
| pitchers | 5 | top50 | 1.686% | 1.549% | 1.698% | 0.00879 | 0.00881 |
| pitchers | 5 | top10 | 0.329% | 0.264% | 0.346% | 0.00234 | 0.00253 |
| pitchers | 7 | top50 | 1.463% | 1.193% | 1.334% | 0.00864 | 0.00867 |
| pitchers | 7 | top10 | 0.278% | 0.172% | 0.233% | 0.00196 | 0.00203 |
| pitchers | 10 | top50 | 0.875% | 0.678% | 0.734% | 0.00546 | 0.00546 |
| pitchers | 10 | top10 | 0.214% | 0.077% | 0.118% | 0.00176 | 0.00168 |

Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):

hitters:

| Group | Predicted | Observed |
|---:|---:|---:|
| 1 | 0.0000 | 0.0000 |
| 2 | 0.0001 | 0.0000 |
| 3 | 0.0001 | 0.0000 |
| 4 | 0.0001 | 0.0000 |
| 5 | 0.0002 | 0.0000 |
| 6 | 0.0004 | 0.0001 |
| 7 | 0.0007 | 0.0002 |
| 8 | 0.0014 | 0.0008 |
| 9 | 0.0052 | 0.0032 |
| 10 | 0.2147 | 0.2335 |

pitchers:

| Group | Predicted | Observed |
|---:|---:|---:|
| 1 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 |
| 6 | 0.0001 | 0.0000 |
| 7 | 0.0001 | 0.0000 |
| 8 | 0.0003 | 0.0000 |
| 9 | 0.0016 | 0.0010 |
| 10 | 0.1818 | 0.1983 |

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38712 | 85% | 0.870 | +26% | 32211 | 0.199 | 0.228 | 0.227 | 82% |
| 2 | 38668 | 75% | 0.875 | +31% | 28648 | 0.199 | 0.226 | 0.224 | 82% |
| 3 | 38638 | 65% | 0.884 | +36% | 24946 | 0.199 | 0.223 | 0.220 | 83% |
| 5 | 38365 | 47% | 0.891 | +38% | 17874 | 0.198 | 0.221 | 0.217 | 82% |
| 7 | 38149 | 32% | 0.890 | +38% | 12092 | 0.197 | 0.215 | 0.210 | 77% |
| 10 | 38660 | 16% | 0.896 | +33% | 6254 | 0.198 | 0.219 | 0.212 | 78% |

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
| 1 | 0.691 | 0.666 | 0.629 |
| 2 | 0.712 | 0.680 | 0.616 |
| 3 | 0.729 | 0.692 | 0.601 |
| 5 | 0.751 | 0.702 | 0.576 |
| 7 | 0.767 | 0.705 | 0.550 |
| 10 | 0.741 | 0.664 | 0.514 |
