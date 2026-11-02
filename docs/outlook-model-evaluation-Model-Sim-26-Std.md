# Outlook model evaluation: Model-Sim-26-Std (Standard26)

Generated 2026-11-02 by `scripts/train_outlook_models.py` from seasons 2037-2099 of the long CPU league (trained in 13 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least five seasons earlier (for horizon 1 the training seasons end 2087, the test seasons are 2093-2098; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 1 | 98125 | 0.967 | 0.906 | 0.961 | 3.82 | 6.31 | 4.19 | +9% / +9% / +8% | 78% | 80% | 49% |
| hitters | 2 | 85909 | 0.916 | 0.647 | 0.896 | 5.57 | 11.63 | 6.30 | +13% / +12% / +10% | 78% | 81% | 48% |
| hitters | 3 | 74455 | 0.853 | 0.187 | 0.816 | 6.76 | 16.48 | 7.70 | +15% / +12% / +13% | 78% | 81% | 49% |
| hitters | 5 | 53263 | 0.733 | -1.222 | 0.639 | 7.98 | 24.25 | 9.54 | +17% / +16% / +19% | 77% | 81% | 47% |
| pitchers | 1 | 26040 | 0.949 | 0.919 | 0.942 | 2.81 | 3.42 | 3.00 | +5% / +6% / +12% | 78% | 79% | 49% |
| pitchers | 2 | 23005 | 0.904 | 0.784 | 0.889 | 3.82 | 5.68 | 4.16 | +6% / +8% / +12% | 78% | 80% | 49% |
| pitchers | 3 | 20092 | 0.852 | 0.561 | 0.823 | 4.55 | 7.99 | 5.02 | +8% / +9% / +12% | 78% | 79% | 48% |
| pitchers | 5 | 14660 | 0.746 | -0.113 | 0.687 | 5.50 | 12.20 | 6.20 | +10% / +11% / +14% | 76% | 80% | 47% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|
| hitters | 16-19 | 16080 | 9.21 | 31.51 | 79% |
| hitters | 20-22 | 21152 | 8.51 | 21.38 | 77% |
| hitters | 23-26 | 21879 | 4.80 | 6.22 | 78% |
| hitters | 27-30 | 11205 | 4.41 | 6.75 | 79% |
| hitters | 31+ | 4139 | 5.01 | 13.56 | 77% |
| pitchers | 16-19 | 3064 | 5.50 | 12.22 | 78% |
| pitchers | 20-22 | 6002 | 5.05 | 10.60 | 79% |
| pitchers | 23-26 | 6322 | 4.52 | 6.15 | 78% |
| pitchers | 27-30 | 2571 | 3.19 | 3.53 | 77% |
| pitchers | 31+ | 2133 | 3.46 | 5.38 | 76% |

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 114759 | 2.477% | 2.399% | 0.997 | 0.969 | +67% | +65% |
| hitters | 1 | top10 | 114759 | 0.514% | 0.471% | 0.999 | 0.963 | +67% | +72% |
| hitters | 2 | top50 | 114391 | 2.446% | 2.343% | 0.994 | 0.949 | +56% | +58% |
| hitters | 2 | top10 | 114391 | 0.511% | 0.421% | 0.998 | 0.956 | +54% | +64% |
| hitters | 3 | top50 | 114184 | 2.378% | 2.264% | 0.990 | 0.916 | +50% | +55% |
| hitters | 3 | top10 | 114184 | 0.501% | 0.390% | 0.997 | 0.930 | +45% | +60% |
| hitters | 5 | top50 | 113821 | 2.106% | 1.910% | 0.985 | 0.838 | +42% | +52% |
| hitters | 5 | top10 | 113821 | 0.448% | 0.285% | 0.994 | 0.850 | +30% | +53% |
| pitchers | 1 | top50 | 30214 | 2.079% | 2.017% | 0.997 | 0.963 | +67% | +65% |
| pitchers | 1 | top10 | 30214 | 0.424% | 0.382% | 0.999 | 0.956 | +67% | +74% |
| pitchers | 2 | top50 | 30210 | 2.046% | 1.944% | 0.995 | 0.958 | +55% | +54% |
| pitchers | 2 | top10 | 30210 | 0.414% | 0.352% | 0.998 | 0.956 | +57% | +66% |
| pitchers | 3 | top50 | 30261 | 1.993% | 1.806% | 0.993 | 0.943 | +49% | +53% |
| pitchers | 3 | top10 | 30261 | 0.397% | 0.320% | 0.998 | 0.951 | +48% | +61% |
| pitchers | 5 | top50 | 29985 | 1.784% | 1.575% | 0.989 | 0.898 | +43% | +53% |
| pitchers | 5 | top10 | 29985 | 0.347% | 0.264% | 0.996 | 0.907 | +37% | +58% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 2.475% | 2.374% | 2.433% | 0.00597 | 0.00596 |
| hitters | 1 | top10 | 0.512% | 0.458% | 0.485% | 0.00165 | 0.00166 |
| hitters | 2 | top50 | 2.425% | 2.275% | 2.350% | 0.00815 | 0.00814 |
| hitters | 2 | top10 | 0.501% | 0.400% | 0.473% | 0.00244 | 0.00247 |
| hitters | 3 | top50 | 2.351% | 2.226% | 2.327% | 0.00988 | 0.00989 |
| hitters | 3 | top10 | 0.497% | 0.389% | 0.503% | 0.00269 | 0.00266 |
| hitters | 5 | top50 | 2.066% | 1.904% | 2.106% | 0.01109 | 0.01109 |
| hitters | 5 | top10 | 0.469% | 0.310% | 0.469% | 0.00325 | 0.00314 |
| pitchers | 1 | top50 | 2.131% | 2.023% | 2.054% | 0.00571 | 0.00580 |
| pitchers | 1 | top10 | 0.440% | 0.386% | 0.431% | 0.00137 | 0.00143 |
| pitchers | 2 | top50 | 2.074% | 1.934% | 1.999% | 0.00753 | 0.00757 |
| pitchers | 2 | top10 | 0.428% | 0.346% | 0.381% | 0.00167 | 0.00170 |
| pitchers | 3 | top50 | 2.004% | 1.780% | 1.920% | 0.00841 | 0.00849 |
| pitchers | 3 | top10 | 0.417% | 0.308% | 0.345% | 0.00218 | 0.00224 |
| pitchers | 5 | top50 | 1.686% | 1.505% | 1.710% | 0.00846 | 0.00841 |
| pitchers | 5 | top10 | 0.329% | 0.253% | 0.348% | 0.00224 | 0.00228 |

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
| 7 | 0.0007 | 0.0003 |
| 8 | 0.0015 | 0.0006 |
| 9 | 0.0053 | 0.0028 |
| 10 | 0.2180 | 0.2339 |

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
| 9 | 0.0015 | 0.0007 |
| 10 | 0.1784 | 0.1986 |

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38712 | 85% | 0.870 | +26% | 32211 | 0.198 | 0.228 | 0.227 | 82% |
| 2 | 38668 | 75% | 0.875 | +31% | 28648 | 0.199 | 0.226 | 0.224 | 82% |
| 3 | 38638 | 65% | 0.884 | +36% | 24946 | 0.199 | 0.223 | 0.220 | 83% |
| 5 | 38365 | 47% | 0.892 | +39% | 17874 | 0.198 | 0.221 | 0.217 | 82% |

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
| 1 | 0.691 | 0.667 | 0.630 |
| 2 | 0.711 | 0.681 | 0.616 |
| 3 | 0.730 | 0.693 | 0.604 |
| 5 | 0.746 | 0.701 | 0.578 |
