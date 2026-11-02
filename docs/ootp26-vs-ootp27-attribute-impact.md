# Attribute impact: Model-Sim-26-Std (OOTP 26) against Model-Sim (OOTP 27)

Generated 2026-11-02 by `scripts/compare_versions.py`. Both sets of models are trained on seasons 2046-2086 of their save; the effects are computed the same way for both (see `docs/model-attribute-ranking.md` for the method).

## How to read it

- **+10 points** is the average change of the predicted value when the rating goes up by 10 points; the **Change** is the 27 effect minus the 26 effect, so a positive change means the rating is worth more in OOTP 27.
- **Share** is the permutation-importance share of the model's skill; a rating that moves together with another shares credit with it, so shares can shift between related ratings without a real change in value.
- **+1 SD** uses one standard deviation of the rating in that save's training data and is the fair way to compare ratings with different spreads.
- **Seed noise** is how much the +1 SD effect of the OOTP 27 model moves when only the cross-validation folds and scrambles change (seed 0 against seed 1). It is a floor: the two simulated leagues also contain different players and seasons, so the true uncertainty of a change is larger. Read changes of about 10-15 % or less, or of a few tenths of a run, as no real difference; the large ones (a rating whose effect doubles or halves) are the findings.
- Units: batting and baserunning runs per 600 PA, pitching runs per 200 IP, fielding runs per season at the position.

## Same ratings, different value

The Model-Sim 2060 ratings scored with both sets of models; the means are weighted by the playing time of that season. `Difference` is the OOTP 27 value minus the OOTP 26 value for the same ratings, in the stream's units. Both are values against the average of the league each model set was trained on, so a mean difference mostly says how good that league's average player is (the OOTP 27 league has better defenders, so the OOTP 26 models see its fielders as above average), not that a player is worth more in one version. Compare the spread (SD) and the rank correlation instead: the correlation says whether the two model sets order the same players the same way, the SD ratio how strongly they separate good from bad.

| Value | Players | Mean, 26 models | Mean, 27 models | Difference | SD, 26 | SD, 27 | Rank correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| baserunning | 417 | +0.49 | +0.24 | -0.26 | 3.35 | 2.57 | 0.97 |
| batting | 417 | -0.17 | +0.61 | +0.79 | 18.21 | 14.70 | 0.99 |
| fielding_2 | 63 | -0.23 | +0.40 | +0.64 | 4.93 | 8.81 | 1.00 |
| fielding_3 | 47 | -0.87 | -0.48 | +0.39 | 4.51 | 3.14 | 0.99 |
| fielding_4 | 62 | +3.41 | +0.39 | -3.02 | 13.15 | 11.76 | 0.97 |
| fielding_5 | 57 | +2.21 | +0.30 | -1.91 | 4.86 | 9.82 | 0.90 |
| fielding_6 | 57 | +7.93 | +0.04 | -7.89 | 15.21 | 13.89 | 0.97 |
| fielding_7 | 51 | -1.82 | +0.60 | +2.41 | 9.97 | 6.97 | 0.80 |
| fielding_8 | 59 | +3.15 | -1.94 | -5.09 | 12.66 | 13.11 | 0.97 |
| fielding_9 | 44 | -0.18 | +0.82 | +1.00 | 10.49 | 6.77 | 0.87 |
| pitching_fip | 442 | -0.67 | +0.81 | +1.48 | 11.12 | 11.69 | 0.97 |
| pitching_babip_runs | 406 | -0.54 | +0.14 | +0.67 | 1.35 | 1.31 | 0.87 |

## Summary: the 25 largest changes of the effect of one standard deviation of a rating

| Rank | Model | Attribute | +1 SD, 26 | +1 SD, 27 | Change | Seed noise |
|---:|---|---|---:|---:|---:|---:|
| 1 | Fielding at C | Catcher framing (`C FRM`) | +2.73 | +8.43 | +5.71 | 0.14 |
| 2 | Fielding at 3B | Infield range (`IF RNG`) | +1.62 | +5.36 | +3.75 | 0.17 |
| 3 | Fielding at CF | Outfield range (`OF RNG`) | +13.16 | +9.83 | -3.33 | 0.67 |
| 4 | Batting | Power (`POW`) | +11.88 | +9.63 | -2.25 | 0.00 |
| 5 | Fielding at 3B | Infield arm (`IF ARM`) | +2.15 | +4.34 | +2.19 | 0.06 |
| 6 | Fielding at RF | Outfield range (`OF RNG`) | +2.96 | +4.80 | +1.83 | 0.11 |
| 7 | Fielding at SS | Infield range (`IF RNG`) | +10.08 | +8.35 | -1.72 | 0.15 |
| 8 | Pitching: FIP part | Home runs allowed (HRA) (`HRR`) | +4.81 | +6.48 | +1.67 | 0.01 |
| 9 | Batting | Avoid strikeouts (AVK) (`K`) | +9.04 | +7.42 | -1.62 | 0.00 |
| 10 | Batting | Eye (walks) (`EYE`) | +5.20 | +3.70 | -1.50 | 0.00 |
| 11 | Fielding at SS | Infield arm (`IF ARM`) | +5.64 | +4.21 | -1.43 | 0.44 |
| 12 | Fielding at LF | Outfield range (`OF RNG`) | +4.21 | +2.83 | -1.38 | 0.12 |
| 13 | Baserunning | Stealing (`STE`) | +3.04 | +1.85 | -1.18 | 0.04 |
| 14 | Fielding at 2B | Infield error (`IF ERR`) | +0.27 | +1.26 | +1.00 | 0.20 |
| 15 | Fielding at 2B | Infield range (`IF RNG`) | +9.11 | +8.23 | -0.88 | 0.09 |
| 16 | Pitching: FIP part | Stamina (`STM`) | -0.41 | -1.21 | -0.80 | 0.03 |
| 17 | Pitching: FIP part | Number of pitches (count) (`PIT`) | -1.03 | -0.25 | +0.78 | 0.00 |
| 18 | Baserunning | Baserunning (`RUN`) | +0.57 | +1.30 | +0.73 | 0.03 |
| 19 | Fielding at 3B | Infield error (`IF ERR`) | +1.02 | +1.73 | +0.71 | 0.04 |
| 20 | Pitching: FIP part | Stuff (`STU`) | +5.62 | +4.91 | -0.71 | 0.01 |
| 21 | Fielding at RF | Outfield arm (`OF ARM`) | +0.62 | -0.08 | -0.71 | 0.03 |
| 22 | Pitching: FIP part | PBABIP (`PBABIP`) | +0.38 | -0.30 | -0.68 | 0.01 |
| 23 | Fielding at LF | Outfield arm (`OF ARM`) | +0.17 | +0.80 | +0.63 | 0.04 |
| 24 | Pitching: FIP part | Control (`CON`) | +3.85 | +3.27 | -0.58 | 0.03 |
| 25 | Batting | Gap power (`GAP`) | +1.92 | +1.44 | -0.48 | 0.00 |

## Batting

Effects in **runs per 600 PA**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.372, Model-Sim 0.403.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Eye (walks) (`EYE`) | +5.40 | +3.49 | -1.91 | -35% | 10% | 7% | +5.20 | +3.70 | -1.50 | 0.00 |
| 2 | BABIP (hit rating) (`BA`) | +6.91 | +5.40 | -1.51 | -22% | 7% | 9% | +4.25 | +4.13 | -0.11 | 0.00 |
| 3 | Power (`POW`) | +11.10 | +9.59 | -1.51 | -14% | 53% | 51% | +11.88 | +9.63 | -2.25 | 0.00 |
| 4 | Gap power (`GAP`) | +2.49 | +1.82 | -0.67 | -27% | 1% | 1% | +1.92 | +1.44 | -0.48 | 0.00 |
| 5 | Avoid strikeouts (AVK) (`K`) | +7.12 | +6.66 | -0.45 | -6% | 29% | 31% | +9.04 | +7.42 | -1.62 | 0.00 |

## Baserunning

Effects in **runs per 600 PA**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.514, Model-Sim 0.505.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Stealing (`STE`) | +1.74 | +1.05 | -0.70 | -40% | 81% | 41% | +3.04 | +1.85 | -1.18 | 0.04 |
| 2 | Baserunning (`RUN`) | +0.34 | +0.86 | +0.53 | +156% | 12% | 51% | +0.57 | +1.30 | +0.73 | 0.03 |
| 3 | Speed (`SPE`) | +0.35 | +0.34 | -0.01 | -4% | 8% | 8% | +0.48 | +0.34 | -0.14 | 0.01 |

## Pitching: FIP part

Effects in **runs per 200 IP**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.260, Model-Sim 0.290.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Home runs allowed (HRA) (`HRR`) | +8.28 | +6.48 | -1.80 | -22% | 47% | 49% | +4.81 | +6.48 | +1.67 | 0.01 |
| 2 | PBABIP (`PBABIP`) | +0.66 | -0.43 | -1.09 | -166% | 1% | 0% | +0.38 | -0.30 | -0.68 | 0.01 |
| 3 | Number of pitches (count) (`PIT`) | -1.03 | -0.25 | +0.78 | -75% | 5% | 1% | -1.03 | -0.25 | +0.78 | 0.00 |
| 4 | Stuff (`STU`) | +5.62 | +4.91 | -0.71 | -13% | 33% | 28% | +5.62 | +4.91 | -0.71 | 0.01 |
| 5 | Control (`CON`) | +3.85 | +3.27 | -0.58 | -15% | 11% | 15% | +3.85 | +3.27 | -0.58 | 0.03 |
| 6 | Stamina (`STM`) | -0.31 | -0.87 | -0.57 | +183% | 3% | 7% | -0.41 | -1.21 | -0.80 | 0.03 |

## Pitching: BABIP part

Effects in **runs saved per 200 IP (a lower BABIP against is better, so a positive effect is good)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.011, Model-Sim 0.005.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Home runs allowed (HRA) (`HRR`) | +0.47 | +0.05 | -0.42 | -89% | 3% | 0% | +0.33 | +0.04 | -0.29 | 0.00 |
| 2 | Control (`CON`) | +0.17 | -0.23 | -0.41 | -232% | 0% | 3% | +0.14 | -0.20 | -0.35 | 0.00 |
| 3 | PBABIP (`PBABIP`) | +2.75 | +2.67 | -0.08 | -3% | 96% | 95% | +1.68 | +1.23 | -0.45 | 0.00 |
| 4 | Stuff (`STU`) | +0.24 | +0.22 | -0.02 | -9% | 1% | 2% | +0.23 | +0.23 | -0.01 | 0.00 |

## Fielding at C

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.576, Model-Sim 0.854.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Catcher framing (`C FRM`) | +5.32 | +8.43 | +3.11 | +58% | 90% | 96% | +2.73 | +8.43 | +5.71 | 0.14 |
| 2 | Catcher ability (blocking) (`C ABI`) | +1.47 | +1.85 | +0.39 | +26% | 10% | 4% | +0.87 | +1.13 | +0.26 | 0.08 |

## Fielding at 1B

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.387, Model-Sim 0.374.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | +2.47 | +1.88 | -0.59 | -24% | 91% | 87% | +2.25 | +1.78 | -0.47 | 0.07 |
| 2 | Height (cm) (`HT`) | +1.31 | +0.94 | -0.38 | -29% | 9% | 13% | +0.88 | +0.74 | -0.14 | 0.00 |

## Fielding at 2B

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.656, Model-Sim 0.723.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Infield error (`IF ERR`) | +0.20 | +1.40 | +1.20 | +613% | 0% | 2% | +0.27 | +1.26 | +1.00 | 0.20 |
| 2 | Infield range (`IF RNG`) | +11.75 | +10.62 | -1.14 | -10% | 94% | 93% | +9.11 | +8.23 | -0.88 | 0.09 |
| 3 | Infield arm (`IF ARM`) | +2.08 | +1.77 | -0.31 | -15% | 6% | 5% | +2.07 | +1.83 | -0.24 | 0.33 |

## Fielding at 3B

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.351, Model-Sim 0.612.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Infield arm (`IF ARM`) | +3.57 | +6.83 | +3.26 | +91% | 58% | 59% | +2.15 | +4.34 | +2.19 | 0.06 |
| 2 | Infield range (`IF RNG`) | +2.72 | +5.36 | +2.64 | +97% | 39% | 37% | +1.62 | +5.36 | +3.75 | 0.17 |
| 3 | Infield error (`IF ERR`) | +1.02 | +1.73 | +0.71 | +70% | 3% | 4% | +1.02 | +1.73 | +0.71 | 0.04 |

## Fielding at SS

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.710, Model-Sim 0.706.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Infield range (`IF RNG`) | +14.22 | +11.23 | -2.99 | -21% | 77% | 76% | +10.08 | +8.35 | -1.72 | 0.15 |
| 2 | Infield arm (`IF ARM`) | +6.44 | +4.80 | -1.64 | -25% | 21% | 22% | +5.64 | +4.21 | -1.43 | 0.44 |
| 3 | Infield error (`IF ERR`) | +2.33 | +2.18 | -0.14 | -6% | 2% | 2% | +1.85 | +1.88 | +0.04 | 0.20 |

## Fielding at LF

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.510, Model-Sim 0.419.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | +5.87 | +2.83 | -3.04 | -52% | 100% | 98% | +4.21 | +2.83 | -1.38 | 0.12 |
| 2 | Outfield arm (`OF ARM`) | +0.43 | +0.80 | +0.38 | +88% | 0% | 2% | +0.17 | +0.80 | +0.63 | 0.04 |

## Fielding at CF

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.619, Model-Sim 0.605.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Outfield range (`OF RNG`) | +17.19 | +13.59 | -3.60 | -21% | 100% | 100% | +13.16 | +9.83 | -3.33 | 0.67 |
| 2 | Outfield arm (`OF ARM`) | +1.02 | +0.78 | -0.24 | -24% | 0% | 0% | +0.80 | +0.71 | -0.09 | 0.01 |

## Fielding at RF

Effects in **runs per season (1200 innings)**. Model fit (player-grouped cross-validated R²): Model-Sim-26-Std 0.637, Model-Sim 0.474.

| Rank | Attribute | +10 points, old | +10 points, new | Change | Change % | Share, old | Share, new | +1 SD, old | +1 SD, new | +1 SD change | Seed noise |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Outfield arm (`OF ARM`) | +0.62 | -0.08 | -0.71 | -114% | 1% | 1% | +0.62 | -0.08 | -0.71 | 0.03 |
| 2 | Outfield range (`OF RNG`) | +4.51 | +4.80 | +0.28 | +6% | 99% | 99% | +2.96 | +4.80 | +1.83 | 0.11 |
