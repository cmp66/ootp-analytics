# Outlook model evaluation: Model-Sim (Standard27)

Generated 2026-11-02 by `scripts/train_outlook_models.py` from seasons 2045-2086 of the long CPU league (trained in 3 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least ten seasons earlier (for horizon 1 the training seasons end 2060, the test seasons are 2071-2076; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 99 | 131303 | 0.871 | 0.644 | 0.796 | 6.71 | 11.21 | 8.38 | +8% / +20% / +26% | 76% | 79% | 47% |
| pitchers | 99 | 49569 | 0.866 | 0.639 | 0.728 | 9.91 | 15.93 | 14.16 | +8% / +30% / +37% | 73% | 81% | 44% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 99 | top50 | 155077 | 2.733% | 2.673% | 0.979 | 0.901 | +47% | +48% |
| hitters | 99 | top10 | 155077 | 0.602% | 0.535% | 0.984 | 0.894 | +45% | +52% |
| pitchers | 99 | top50 | 59419 | 1.967% | 1.942% | 0.972 | 0.901 | +41% | +42% |
| pitchers | 99 | top10 | 59419 | 0.402% | 0.407% | 0.962 | 0.872 | +29% | +39% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 99 | top50 | 2.768% | 2.715% | 2.782% | 0.01157 | 0.01158 |
| hitters | 99 | top10 | 0.622% | 0.522% | 0.552% | 0.00331 | 0.00332 |
| pitchers | 99 | top50 | 1.973% | 1.922% | 1.921% | 0.01020 | 0.01023 |
| pitchers | 99 | top10 | 0.426% | 0.406% | 0.366% | 0.00321 | 0.00315 |

Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
