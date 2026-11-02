# Which ratings matter: ranking of the model attributes

Generated 2026-11-02 by `scripts/rating_importance.py` from the models trained on **Model-Sim** (ratings set `Standard27`). Regenerate after retraining.

## How to read it

- **Share of the model's skill** is the loss in held-out R² when the rating is scrambled (permutation importance on
  players the model has not seen, five player-grouped folds), as a share of the total over the ratings of the model.
  Ratings that move together (for example range and the composite position rating) share their credit.
- **Effect of +10 points** is the average change of the predicted value when the rating goes up by 10 points (two
  steps on the 20-80 scale) with everything else unchanged. **Effect of +1 SD** raises the rating by one standard
  deviation of that rating among the players of the model (the SD is in brackets), which makes ratings with a
  different spread comparable. Where a model is non-linear (boosting, network) the effect is an average.
- 'R² lost when scrambled' can exceed 1: a scrambled rating that the model relies on gives predictions worse than
  the average, so R² goes negative.
- Units differ by stream: batting and baserunning are runs per 600 PA, pitching runs per 200 IP, fielding runs per
  full season at the position. A regular's 600 PA, a starter's 200 IP and a fielder's 1200 innings are each about a
  season, so the summary table compares the +1 SD effects across streams as seasonal runs; a reliever throws about
  a third of 200 IP, so his seasonal effects are about a third of the pitching figures.
- Batting and pitching come from the stacked platoon models: BABIP (`BA`), gap power, power, eye and AVK are the
  ratings against the pitcher (batter) of the row, vs right and vs left combined in one attribute.
- `CON` and `MOV` are derived ratings (see `docs/derived-ratings.md`) and are not in the models; contact is half
  BABIP and half AVK, movement mostly home runs allowed.
- The ranking is for the Standard27 rating scale of this save; a save with a different scale needs its own models and
  its own ranking. Ratings that are not listed for a stream had no measurable value in the attribute review
  (`plans/14-models.md`).

## Summary: the 20 biggest effects of one standard deviation of a rating on seasonal value

| Rank | Stream | Attribute | Effect of +1 SD (runs per season) | Share within its model |
|---:|---|---|---:|---:|
| 1 | Fielding at CF | Outfield range (`OF RNG`) | +9.83 | 100% |
| 2 | Batting | Power (`POW`) | +9.63 | 51% |
| 3 | Fielding at C | Catcher framing (`C FRM`) | +8.43 | 96% |
| 4 | Fielding at SS | Infield range (`IF RNG`) | +8.35 | 76% |
| 5 | Fielding at 2B | Infield range (`IF RNG`) | +8.23 | 93% |
| 6 | Batting | Avoid strikeouts (AVK) (`K`) | +7.42 | 31% |
| 7 | Pitching: FIP part (strikeouts, walks, home runs) | Home runs allowed (HRA) (`HRR`) | +6.48 | 49% |
| 8 | Fielding at 3B | Infield range (`IF RNG`) | +5.36 | 37% |
| 9 | Pitching: FIP part (strikeouts, walks, home runs) | Stuff (`STU`) | +4.91 | 28% |
| 10 | Fielding at RF | Outfield range (`OF RNG`) | +4.80 | 99% |
| 11 | Fielding at 3B | Infield arm (`IF ARM`) | +4.34 | 59% |
| 12 | Fielding at SS | Infield arm (`IF ARM`) | +4.21 | 22% |
| 13 | Batting | BABIP (hit rating) (`BA`) | +4.13 | 9% |
| 14 | Batting | Eye (walks) (`EYE`) | +3.70 | 7% |
| 15 | Pitching: FIP part (strikeouts, walks, home runs) | Control (`CON`) | +3.27 | 15% |
| 16 | Fielding at LF | Outfield range (`OF RNG`) | +2.83 | 98% |
| 17 | Fielding at SS | Infield error (`IF ERR`) | +1.88 | 2% |
| 18 | Baserunning | Stealing (`STE`) | +1.85 | 41% |
| 19 | Fielding at 2B | Infield arm (`IF ARM`) | +1.83 | 5% |
| 20 | Fielding at 1B | Infield range (`IF RNG`) | +1.78 | 87% |

## What the ranking says (written by hand for the Model-Sim run of 2026-11-02; check it after retraining)

- **Batting:** power (about half of the model's skill) and avoiding strikeouts (about a third) carry most of the value;
  BABIP and eye are minor and gap power adds almost nothing once the others are known. +10 points of power is worth
  about 9.6 runs per 600 PA, strikeout avoidance 6.7, BABIP 5.4, eye 3.5 and gap 1.8.
- **Baserunning:** baserunning ability (`RUN`) and stealing (`STE`) are 92 % of it; raw speed adds little once those
  are known (+10 points: 0.9, 1.1 and 0.3 runs per 600 PA).
- **Pitching:** home runs allowed (49 %) leads, then stuff (28 %) and control (15 %): +10 points are worth 6.5, 4.9 and
  3.3 runs per 200 IP. PBABIP acts through the BABIP part, where +10 points is worth about 2.7 runs per 200 IP.
  Stamina and pitch count mostly mark the pitcher's role (starters average 55 stamina, relievers 38) and the value is
  measured against his own usage group, so their small negative effects are role corrections, not penalties: among
  starters they have no effect, among relievers a higher stamina is slightly worse (-0.17 runs per 200 IP per point).
- **Fielding:** range is nearly everything at the outfield positions, 2B and 1B (75 % to 100 %). Infield arm matters most
  at third base (59 % against 37 % for range), then shortstop (22 %) and little at second (5 %); outfield arm is worth
  almost nothing (under 2 %); error rating 2 % to 4 % for infielders. Catcher framing is 96 % of the catcher's value.
  +10 points of range: CF 13.6, SS 11.2, 2B 10.6, 3B 5.4, RF 4.8, LF 2.8, 1B 1.9 runs per season. First basemen get
  13 % of their model from height.

## Batting

Model `batting-split` (ridge), 36227 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.403.
Effects are in **runs per 600 PA**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Power (`POW`) | 51% | 0.378 | +9.59 (+10 points) | +9.63 (10.0) |
| 2 | Avoid strikeouts (AVK) (`K`) | 31% | 0.226 | +6.66 (+10 points) | +7.42 (11.1) |
| 3 | BABIP (hit rating) (`BA`) | 9% | 0.070 | +5.40 (+10 points) | +4.13 (7.7) |
| 4 | Eye (walks) (`EYE`) | 7% | 0.054 | +3.49 (+10 points) | +3.70 (10.6) |
| 5 | Gap power (`GAP`) | 1% | 0.008 | +1.82 (+10 points) | +1.44 (7.9) |

## Baserunning

Model `baserunning` (boosting), 17214 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.505.
Effects are in **runs per 600 PA**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Baserunning (`RUN`) | 51% | 0.255 | +0.86 (+10 points) | +1.30 (13.4) |
| 2 | Stealing (`STE`) | 41% | 0.206 | +1.05 (+10 points) | +1.85 (13.3) |
| 3 | Speed (`SPE`) | 8% | 0.040 | +0.34 (+10 points) | +0.34 (12.3) |

## Pitching: FIP part (strikeouts, walks, home runs)

Model `pitching-fip-split` (boosting), 38417 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.290.
Effects are in **runs per 200 IP**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Home runs allowed (HRA) (`HRR`) | 49% | 0.237 | +6.48 (+10 points) | +6.48 (8.6) |
| 2 | Stuff (`STU`) | 28% | 0.134 | +4.91 (+10 points) | +4.91 (10.8) |
| 3 | Control (`CON`) | 15% | 0.074 | +3.27 (+10 points) | +3.27 (8.9) |
| 4 | Stamina (`STM`) | 7% | 0.033 | -0.87 (+10 points) | -1.21 (14.9) |
| 5 | Number of pitches (count) (`PIT`) | 1% | 0.003 | -0.25 (+1 pitch) | -0.25 (1.2) |
| 6 | PBABIP (`PBABIP`) | 0% | 0.001 | -0.43 (+10 points) | -0.30 (4.7) |

## Pitching: BABIP part (hits on balls in play)

Model `pitching-babip-split` (ridge), 33772 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.005.
Effects are in **runs saved per 200 IP (a lower BABIP against is better, so a positive effect is good)**.

This model explains almost none of the variance (almost all of a pitcher's BABIP is luck), so its ranking is not informative; the effect of the rating is. A BABIP difference of 0.001 over 200 innings is about 0.44 runs (563 balls in play x 0.79 runs per hit).

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | PBABIP (`PBABIP`) | 95% | 0.010 | +2.67 (+10 points) | +1.23 (4.6) |
| 2 | Control (`CON`) | 3% | 0.000 | -0.23 (+10 points) | -0.20 (8.9) |
| 3 | Stuff (`STU`) | 2% | 0.000 | +0.22 (+10 points) | +0.23 (10.4) |
| 4 | Home runs allowed (HRA) (`HRR`) | 0% | -0.000 | +0.05 (+10 points) | +0.04 (8.6) |

## Fielding at C

Model `fielding-2` (boosting), 2543 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.854.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Catcher framing (`C FRM`) | 96% | 1.488 | +8.43 (+10 points) | +8.43 (7.6) |
| 2 | Catcher ability (blocking) (`C ABI`) | 4% | 0.069 | +1.85 (+10 points) | +1.13 (7.4) |

## Fielding at 1B

Model `fielding-3` (mlp), 2043 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.374.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | 87% | 0.725 | +1.88 (+10 points) | +1.78 (9.1) |
| 2 | Height (cm) (`HT`) | 13% | 0.113 | +0.94 (+10 cm) | +0.74 (4.9) |

## Fielding at 2B

Model `fielding-4` (mlp), 2164 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.723.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | 93% | 1.281 | +10.62 (+10 points) | +8.23 (6.8) |
| 2 | Infield arm (`IF ARM`) | 5% | 0.070 | +1.77 (+10 points) | +1.83 (10.4) |
| 3 | Infield error (`IF ERR`) | 2% | 0.026 | +1.40 (+10 points) | +1.26 (8.4) |

## Fielding at 3B

Model `fielding-5` (boosting), 2200 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.612.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield arm (`IF ARM`) | 59% | 0.882 | +6.83 (+10 points) | +4.34 (7.1) |
| 2 | Infield range (`IF RNG`) | 37% | 0.561 | +5.36 (+10 points) | +5.36 (7.7) |
| 3 | Infield error (`IF ERR`) | 4% | 0.054 | +1.73 (+10 points) | +1.73 (8.6) |

## Fielding at SS

Model `fielding-6` (mlp), 2179 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.706.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | 76% | 0.905 | +11.23 (+10 points) | +8.35 (6.7) |
| 2 | Infield arm (`IF ARM`) | 22% | 0.265 | +4.80 (+10 points) | +4.21 (8.5) |
| 3 | Infield error (`IF ERR`) | 2% | 0.028 | +2.18 (+10 points) | +1.88 (8.7) |

## Fielding at LF

Model `fielding-7` (boosting), 2312 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.419.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | 98% | 0.941 | +2.83 (+10 points) | +2.83 (7.5) |
| 2 | Outfield arm (`OF ARM`) | 2% | 0.018 | +0.80 (+10 points) | +0.80 (8.1) |

## Fielding at CF

Model `fielding-8` (mlp), 2062 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.605.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | 100% | 1.290 | +13.59 (+10 points) | +9.83 (5.3) |
| 2 | Outfield arm (`OF ARM`) | 0% | 0.005 | +0.78 (+10 points) | +0.71 (8.7) |

## Fielding at RF

Model `fielding-9` (boosting), 2185 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.474.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | 99% | 0.996 | +4.80 (+10 points) | +4.80 (7.5) |
| 2 | Outfield arm (`OF ARM`) | 1% | 0.006 | -0.08 (+10 points) | -0.08 (8.8) |
