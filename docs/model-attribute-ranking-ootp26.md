# Which ratings matter: ranking of the model attributes

Generated 2026-11-02 by `scripts/rating_importance.py` from the models trained on **Model-Sim-26-Std** (ratings set `Standard26`). Regenerate after retraining.

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
- The ranking is for the Standard26 rating scale of this save; a save with a different scale needs its own models and
  its own ranking. Ratings that are not listed for a stream had no measurable value in the attribute review
  (`plans/14-models.md`).

## Summary: the 20 biggest effects of one standard deviation of a rating on seasonal value

| Rank | Stream | Attribute | Effect of +1 SD (runs per season) | Share within its model |
|---:|---|---|---:|---:|
| 1 | Fielding at CF | Outfield range (`OF RNG`) | +13.16 | 100% |
| 2 | Batting | Power (`POW`) | +11.88 | 53% |
| 3 | Fielding at SS | Infield range (`IF RNG`) | +10.08 | 77% |
| 4 | Fielding at 2B | Infield range (`IF RNG`) | +9.11 | 94% |
| 5 | Batting | Avoid strikeouts (AVK) (`K`) | +9.04 | 29% |
| 6 | Fielding at SS | Infield arm (`IF ARM`) | +5.64 | 21% |
| 7 | Pitching: FIP part (strikeouts, walks, home runs) | Stuff (`STU`) | +5.62 | 33% |
| 8 | Batting | Eye (walks) (`EYE`) | +5.20 | 10% |
| 9 | Pitching: FIP part (strikeouts, walks, home runs) | Home runs allowed (HRA) (`HRR`) | +4.81 | 47% |
| 10 | Batting | BABIP (hit rating) (`BA`) | +4.25 | 7% |
| 11 | Fielding at LF | Outfield range (`OF RNG`) | +4.21 | 100% |
| 12 | Pitching: FIP part (strikeouts, walks, home runs) | Control (`CON`) | +3.85 | 11% |
| 13 | Baserunning | Stealing (`STE`) | +3.04 | 81% |
| 14 | Fielding at RF | Outfield range (`OF RNG`) | +2.96 | 99% |
| 15 | Fielding at C | Catcher framing (`C FRM`) | +2.73 | 90% |
| 16 | Fielding at 1B | Infield range (`IF RNG`) | +2.25 | 91% |
| 17 | Fielding at 3B | Infield arm (`IF ARM`) | +2.15 | 58% |
| 18 | Fielding at 2B | Infield arm (`IF ARM`) | +2.07 | 6% |
| 19 | Batting | Gap power (`GAP`) | +1.92 | 1% |
| 20 | Fielding at SS | Infield error (`IF ERR`) | +1.85 | 2% |

## What the ranking says (written by hand for the Model-Sim-26-Std run of 2026-11-02; check it after retraining)

The OOTP 26 ranking follows the OOTP 27 one (`model-attribute-ranking.md`) in most places. The differences are listed last.

- **Batting:** power (53 % of the model's skill) and avoiding strikeouts (29 %) carry most of the value; eye is 10 %, BABIP
  7 % and gap power 1 %. +10 points of power is worth about 11.1 runs per 600 PA, strikeout avoidance 7.1, BABIP 6.9,
  eye 5.4 and gap 2.5.
- **Baserunning:** stealing (`STE`) is 81 % of it (+10 points: 1.7 runs per 600 PA); baserunning (`RUN`) and speed
  (`SPE`) are 12 % and 8 % (0.3 to 0.4 runs per +10 points).
- **Pitching:** home runs allowed (47 %) leads, then stuff (33 %) and control (11 %): +10 points are worth 8.3, 5.6 and
  3.9 runs per 200 IP. PBABIP acts through the BABIP part, where +10 points is worth about 2.7 runs per 200 IP. Pitch
  count (-1.0 runs per pitch) and stamina (-0.3 per +10 points) are small negative effects; as in OOTP 27 they most likely
  mark the pitcher's role (value is measured against his own usage group) rather than penalise him, but this has not
  been checked per usage group for OOTP 26.
- **Fielding:** range is nearly everything at the outfield positions, 2B and 1B (91 % to 100 %). Infield arm matters most
  at third base (58 % against 39 % for range), then shortstop (21 %), little at second (6 %); outfield arm is worth
  about 1 % or less; error rating 0 % to 3 % for infielders. Catcher framing is 90 % of the catcher's value, blocking
  (`C ABI`) the rest. First basemen get 9 % of their model from height.
  +10 points of range: CF 17.2, SS 14.2, 2B 11.8, LF 5.9, RF 4.5, 3B 2.7, 1B 2.5 runs per season.

**Compared with OOTP 27**
- Batting ratings are worth more per point in OOTP 26 (power 11.1 against 9.6, BABIP 6.9 against 5.4, eye 5.4 against
  3.5 runs per +10 points).
- Baserunning: stealing carries more (81 % against a share of 92 % for `RUN` and `STE` together in OOTP 27, where `RUN`
  and `STE` were of similar size); `RUN` is now small (12 %).
- Pitching: stuff and home runs allowed are worth more per point (5.6 and 8.3 against 4.9 and 6.5 runs per +10
  points), control is a little more (3.9 against 3.3).
- Fielding: range at third base is worth half as much per point (2.7 against 5.4 runs), at left field twice as much (5.9
  against 2.8), at center field and shortstop about a quarter more. The 3B, 1B and C models fit worse in OOTP 26 (R² 0.35,
  0.39 and 0.58 against 0.61, 0.37 and 0.85) and the catcher model depends a little less on framing.

## Batting

Model `batting-split` (ridge), 41338 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.372.
Effects are in **runs per 600 PA**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Power (`POW`) | 53% | 0.530 | +11.10 (+10 points) | +11.88 (10.7) |
| 2 | Avoid strikeouts (AVK) (`K`) | 29% | 0.295 | +7.12 (+10 points) | +9.04 (12.7) |
| 3 | Eye (walks) (`EYE`) | 10% | 0.101 | +5.40 (+10 points) | +5.20 (9.6) |
| 4 | BABIP (hit rating) (`BA`) | 7% | 0.068 | +6.91 (+10 points) | +4.25 (6.1) |
| 5 | Gap power (`GAP`) | 1% | 0.014 | +2.49 (+10 points) | +1.92 (7.7) |

## Baserunning

Model `baserunning` (boosting), 19764 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.514.
Effects are in **runs per 600 PA**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Stealing (`STE`) | 81% | 0.487 | +1.74 (+10 points) | +3.04 (15.9) |
| 2 | Baserunning (`RUN`) | 12% | 0.070 | +0.34 (+10 points) | +0.57 (19.1) |
| 3 | Speed (`SPE`) | 8% | 0.046 | +0.35 (+10 points) | +0.48 (16.2) |

## Pitching: FIP part (strikeouts, walks, home runs)

Model `pitching-fip-split` (boosting), 41686 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.260.
Effects are in **runs per 200 IP**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Home runs allowed (HRA) (`HRR`) | 47% | 0.189 | +8.28 (+10 points) | +4.81 (7.1) |
| 2 | Stuff (`STU`) | 33% | 0.133 | +5.62 (+10 points) | +5.62 (10.2) |
| 3 | Control (`CON`) | 11% | 0.046 | +3.85 (+10 points) | +3.85 (8.1) |
| 4 | Number of pitches (count) (`PIT`) | 5% | 0.021 | -1.03 (+1 pitch) | -1.03 (1.1) |
| 5 | Stamina (`STM`) | 3% | 0.013 | -0.31 (+10 points) | -0.41 (14.5) |
| 6 | PBABIP (`PBABIP`) | 1% | 0.003 | +0.66 (+10 points) | +0.38 (6.1) |

## Pitching: BABIP part (hits on balls in play)

Model `pitching-babip-split` (ridge), 37756 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.011.
Effects are in **runs saved per 200 IP (a lower BABIP against is better, so a positive effect is good)**.

This model explains almost none of the variance (almost all of a pitcher's BABIP is luck), so its ranking is not informative; the effect of the rating is. A BABIP difference of 0.001 over 200 innings is about 0.46 runs (572 balls in play x 0.81 runs per hit).

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | PBABIP (`PBABIP`) | 96% | 0.017 | +2.75 (+10 points) | +1.68 (6.1) |
| 2 | Home runs allowed (HRA) (`HRR`) | 3% | 0.000 | +0.47 (+10 points) | +0.33 (7.0) |
| 3 | Stuff (`STU`) | 1% | 0.000 | +0.24 (+10 points) | +0.23 (9.9) |
| 4 | Control (`CON`) | 0% | 0.000 | +0.17 (+10 points) | +0.14 (8.1) |

## Fielding at C

Model `fielding-2` (boosting), 2755 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.576.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Catcher framing (`C FRM`) | 90% | 0.809 | +5.32 (+10 points) | +2.73 (6.4) |
| 2 | Catcher ability (blocking) (`C ABI`) | 10% | 0.086 | +1.47 (+10 points) | +0.87 (6.4) |

## Fielding at 1B

Model `fielding-3` (mlp), 2652 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.387.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | 91% | 0.746 | +2.47 (+10 points) | +2.25 (8.6) |
| 2 | Height (cm) (`HT`) | 9% | 0.078 | +1.31 (+10 cm) | +0.88 (4.6) |

## Fielding at 2B

Model `fielding-4` (mlp), 2618 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.656.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | 94% | 1.257 | +11.75 (+10 points) | +9.11 (6.5) |
| 2 | Infield arm (`IF ARM`) | 6% | 0.084 | +2.08 (+10 points) | +2.07 (9.9) |
| 3 | Infield error (`IF ERR`) | 0% | -0.000 | +0.20 (+10 points) | +0.27 (7.2) |

## Fielding at 3B

Model `fielding-5` (boosting), 2572 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.351.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield arm (`IF ARM`) | 58% | 0.472 | +3.57 (+10 points) | +2.15 (7.0) |
| 2 | Infield range (`IF RNG`) | 39% | 0.320 | +2.72 (+10 points) | +1.62 (7.2) |
| 3 | Infield error (`IF ERR`) | 3% | 0.027 | +1.02 (+10 points) | +1.02 (7.8) |

## Fielding at SS

Model `fielding-6` (mlp), 2524 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.710.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | 77% | 0.893 | +14.22 (+10 points) | +10.08 (6.7) |
| 2 | Infield arm (`IF ARM`) | 21% | 0.243 | +6.44 (+10 points) | +5.64 (8.6) |
| 3 | Infield error (`IF ERR`) | 2% | 0.026 | +2.33 (+10 points) | +1.85 (7.9) |

## Fielding at LF

Model `fielding-7` (boosting), 2695 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.510.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | 100% | 1.144 | +5.87 (+10 points) | +4.21 (5.4) |
| 2 | Outfield arm (`OF ARM`) | 0% | 0.004 | +0.43 (+10 points) | +0.17 (7.4) |

## Fielding at CF

Model `fielding-8` (mlp), 2635 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.619.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | 100% | 1.237 | +17.19 (+10 points) | +13.16 (6.8) |
| 2 | Outfield arm (`OF ARM`) | 0% | 0.000 | +1.02 (+10 points) | +0.80 (8.1) |

## Fielding at RF

Model `fielding-9` (boosting), 2564 rows, trained on seasons 2046-2086, player-grouped cross-validated R² 0.637.
Effects are in **runs per season (1200 innings)**.

| Rank | Attribute | Share of the model's skill | R² lost when scrambled | Effect of +10 points | Effect of +1 SD (SD) |
|---:|---|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | 99% | 1.493 | +4.51 (+10 points) | +2.96 (7.1) |
| 2 | Outfield arm (`OF ARM`) | 1% | 0.008 | +0.62 (+10 points) | +0.62 (7.6) |
