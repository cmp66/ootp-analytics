# Outlook model evaluation: Model-Sim-26-Std (Standard26)

Generated 2026-11-06 by `scripts/train_outlook_models.py` from seasons 2037-2099 of the long CPU league (trained in 19 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least ten seasons earlier (for horizon 1 the training seasons end 2082, the test seasons are 2093-2098; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 1 | 98125 | 0.968 | 0.906 | 0.961 | 3.76 | 6.31 | 4.19 | +9% / +10% / +9% | 78% | 80% | 49% |
| hitters | 5 | 53263 | 0.743 | -1.222 | 0.639 | 7.82 | 24.25 | 9.54 | +18% / +18% / +20% | 77% | 81% | 48% |
| hitters | 7 | 36275 | 0.648 | -3.091 | 0.434 | 8.09 | 29.32 | 10.63 | +25% / +24% / +25% | 75% | 80% | 45% |
| hitters | 10 | 19508 | 0.639 | -4.367 | 0.263 | 8.16 | 33.91 | 12.17 | +34% / +33% / +31% | 71% | 81% | 42% |
| hitters | 99 | 101276 | 0.808 | -0.203 | 0.738 | 7.15 | 17.20 | 8.31 | +10% / +14% / +19% | 77% | 80% | 48% |
| pitchers | 1 | 26040 | 0.949 | 0.919 | 0.942 | 2.81 | 3.42 | 3.00 | +5% / +6% / +12% | 78% | 78% | 49% |
| pitchers | 5 | 14660 | 0.747 | -0.113 | 0.687 | 5.47 | 12.20 | 6.20 | +10% / +12% / +14% | 76% | 79% | 47% |
| pitchers | 7 | 10045 | 0.638 | -1.166 | 0.535 | 5.89 | 15.43 | 6.83 | +12% / +14% / +15% | 74% | 79% | 45% |
| pitchers | 10 | 5013 | 0.475 | -3.121 | 0.161 | 5.60 | 16.47 | 7.40 | +23% / +24% / +20% | 72% | 79% | 43% |
| pitchers | 99 | 26478 | 0.815 | 0.210 | 0.747 | 4.84 | 9.51 | 5.62 | +12% / +14% / +15% | 77% | 78% | 48% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 114759 | 2.477% | 2.408% | 0.997 | 0.969 | +67% | +65% |
| hitters | 1 | top10 | 114759 | 0.514% | 0.467% | 0.999 | 0.963 | +67% | +72% |
| hitters | 5 | top50 | 113821 | 2.106% | 1.937% | 0.985 | 0.838 | +42% | +53% |
| hitters | 5 | top10 | 113821 | 0.448% | 0.310% | 0.994 | 0.850 | +31% | +54% |
| hitters | 7 | top50 | 113684 | 1.688% | 1.554% | 0.984 | 0.795 | +37% | +53% |
| hitters | 7 | top10 | 113684 | 0.341% | 0.237% | 0.992 | 0.813 | +23% | +51% |
| hitters | 10 | top50 | 115083 | 0.964% | 0.843% | 0.988 | 0.725 | +33% | +56% |
| hitters | 10 | top10 | 115083 | 0.189% | 0.130% | 0.994 | 0.700 | +17% | +54% |
| hitters | 99 | top50 | 115083 | 4.113% | 4.244% | 0.987 | 0.862 | +54% | +58% |
| hitters | 99 | top10 | 115083 | 1.011% | 0.983% | 0.995 | 0.868 | +54% | +65% |
| pitchers | 1 | top50 | 30214 | 2.079% | 2.027% | 0.997 | 0.963 | +67% | +65% |
| pitchers | 1 | top10 | 30214 | 0.424% | 0.386% | 0.999 | 0.956 | +66% | +74% |
| pitchers | 5 | top50 | 29985 | 1.784% | 1.625% | 0.988 | 0.898 | +42% | +52% |
| pitchers | 5 | top10 | 29985 | 0.347% | 0.259% | 0.996 | 0.907 | +32% | +55% |
| pitchers | 7 | top50 | 29735 | 1.423% | 1.221% | 0.987 | 0.840 | +37% | +53% |
| pitchers | 7 | top10 | 29735 | 0.276% | 0.169% | 0.996 | 0.869 | +30% | +58% |
| pitchers | 10 | top50 | 30183 | 0.795% | 0.672% | 0.990 | 0.783 | +32% | +55% |
| pitchers | 10 | top10 | 30183 | 0.192% | 0.086% | 0.996 | 0.806 | +19% | +52% |
| pitchers | 99 | top50 | 30183 | 3.485% | 3.479% | 0.989 | 0.892 | +56% | +59% |
| pitchers | 99 | top10 | 30183 | 0.914% | 0.687% | 0.991 | 0.889 | +44% | +54% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 1 | top50 | 2.475% | 2.392% | 2.447% | 0.00596 | 0.00601 |
| hitters | 1 | top10 | 0.512% | 0.453% | 0.485% | 0.00166 | 0.00165 |
| hitters | 5 | top50 | 2.066% | 1.922% | 2.094% | 0.01114 | 0.01110 |
| hitters | 5 | top10 | 0.469% | 0.326% | 0.458% | 0.00326 | 0.00322 |
| hitters | 7 | top50 | 1.658% | 1.513% | 1.625% | 0.01044 | 0.01051 |
| hitters | 7 | top10 | 0.337% | 0.238% | 0.351% | 0.00261 | 0.00262 |
| hitters | 10 | top50 | 0.895% | 0.768% | 0.873% | 0.00629 | 0.00627 |
| hitters | 10 | top10 | 0.154% | 0.100% | 0.159% | 0.00137 | 0.00141 |
| hitters | 99 | top50 | 4.058% | 4.203% | 4.094% | 0.01479 | 0.01492 |
| hitters | 99 | top10 | 0.967% | 0.931% | 0.950% | 0.00404 | 0.00404 |
| pitchers | 1 | top50 | 2.131% | 2.029% | 2.040% | 0.00571 | 0.00581 |
| pitchers | 1 | top10 | 0.440% | 0.399% | 0.448% | 0.00141 | 0.00150 |
| pitchers | 5 | top50 | 1.686% | 1.550% | 1.717% | 0.00886 | 0.00885 |
| pitchers | 5 | top10 | 0.329% | 0.256% | 0.346% | 0.00242 | 0.00258 |
| pitchers | 7 | top50 | 1.463% | 1.195% | 1.328% | 0.00872 | 0.00886 |
| pitchers | 7 | top10 | 0.278% | 0.165% | 0.239% | 0.00212 | 0.00227 |
| pitchers | 10 | top50 | 0.875% | 0.676% | 0.720% | 0.00556 | 0.00555 |
| pitchers | 10 | top10 | 0.214% | 0.073% | 0.131% | 0.00186 | 0.00184 |
| pitchers | 99 | top50 | 3.554% | 3.433% | 3.303% | 0.01255 | 0.01296 |
| pitchers | 99 | top10 | 0.915% | 0.648% | 0.811% | 0.00470 | 0.00457 |

Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38712 | 85% | 0.871 | +26% | 32211 | 0.198 | 0.228 | 0.227 | 81% |
| 5 | 38365 | 47% | 0.894 | +39% | 17874 | 0.198 | 0.221 | 0.217 | 82% |
| 7 | 38149 | 32% | 0.893 | +39% | 12092 | 0.197 | 0.215 | 0.210 | 77% |
| 10 | 38660 | 16% | 0.901 | +35% | 6254 | 0.198 | 0.219 | 0.212 | 78% |

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
| 1 | 0.689 | 0.666 | 0.629 |
| 5 | 0.747 | 0.702 | 0.577 |
| 7 | 0.764 | 0.705 | 0.550 |
| 10 | 0.741 | 0.664 | 0.515 |
