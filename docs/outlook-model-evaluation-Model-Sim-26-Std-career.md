# Outlook model evaluation: Model-Sim-26-Std (Standard26)

Generated 2026-11-02 by `scripts/train_outlook_models.py` from seasons 2037-2099 of the long CPU league (trained in 3 minutes). Each game version has its own models.

Hold-out: the test rows are the latest six seasons with an observed future; training uses rows at least ten seasons earlier (for horizon 1 the training seasons end 2073, the test seasons are 2084-2089; the dates move with the horizon). The final models are retrained on all observed rows.

## Value: where the rating-based value of a slot is going

Quantile models of the change of the slot value h seasons ahead, for players ranked at the slot in both seasons. R² is
of the value h seasons ahead. Baselines: persistence (value stays) and the median change of the same slot and age band.
Pinball skill is the gain in pinball loss over the age-band baseline for the 10th, 50th and 90th percentile.
The 80 % band is [10th, 90th]; calibrated = widened by the shift fitted on the first half of the test seasons and
checked on the second half.

| Family | h | Test rows | R² | Persistence | Age baseline | MAE | MAE persistence | MAE age baseline | Skill q10 / q50 / q90 | 80 % band | calibrated | 50 % band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| hitters | 99 | 101276 | 0.799 | -0.203 | 0.738 | 7.34 | 17.20 | 8.31 | +9% / +12% / +16% | 77% | 80% | 47% |
| pitchers | 99 | 26478 | 0.812 | 0.210 | 0.747 | 4.88 | 9.51 | 5.62 | +12% / +13% / +14% | 77% | 79% | 48% |

By age band (horizon 3): median error against persistence, and the raw 80 % band coverage:

| Family | Age | Rows | MAE | MAE persistence | 80 % band |
|---|---|---:|---:|---:|---:|

## Tier chances: top 10 and top 50 at the slot

Classifiers for being ranked top 10 / top 50 at the same slot h seasons ahead (a player who is gone counts as no). The
baseline is the observed rate by slot and decile of today's percentile. Brier skill is the gain over that baseline.

| Family | h | Tier | Test rows | Base rate | Mean prediction | AUC | AUC baseline | Brier skill | Log-loss skill |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hitters | 99 | top50 | 115083 | 4.113% | 4.235% | 0.987 | 0.862 | +54% | +58% |
| hitters | 99 | top10 | 115083 | 1.011% | 0.976% | 0.995 | 0.868 | +53% | +64% |
| pitchers | 99 | top50 | 30183 | 3.485% | 3.456% | 0.989 | 0.892 | +55% | +58% |
| pitchers | 99 | top10 | 30183 | 0.914% | 0.703% | 0.991 | 0.889 | +44% | +54% |

The chances are a little low in absolute terms in the latest seasons (the share of future top players rises over time),
so the stored classifiers carry an isotonic calibration fitted on the test rows. Check: fitted on the first half of the
test seasons, scored on the second half:

| Family | h | Tier | Observed rate | Mean raw | Mean calibrated | Brier raw | Brier calibrated |
|---|---:|---|---:|---:|---:|---:|---:|
| hitters | 99 | top50 | 4.058% | 4.191% | 4.087% | 0.01495 | 0.01505 |
| hitters | 99 | top10 | 0.967% | 0.928% | 0.953% | 0.00411 | 0.00415 |
| pitchers | 99 | top50 | 3.554% | 3.405% | 3.287% | 0.01273 | 0.01320 |
| pitchers | 99 | top10 | 0.915% | 0.667% | 0.821% | 0.00462 | 0.00447 |

Calibration of the top-50 chance at horizon 3 (rows sorted by prediction, ten equal groups):

## Staying in the league and availability

| h | Stay: rows | Base rate | AUC | Brier skill vs age | Availability: rows | MAE | MAE constant | MAE tier median | 80 % band |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Median availability predicted for players of each proneness tier (0 durable, 1 normal, 2 fragile; test seasons):

| h | Durable | Normal | Fragile |
|---:|---:|---:|---:|
