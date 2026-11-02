# Derived ratings: CON and MOV

Two ratings are not independent skills: OOTP computes them from other ratings. They carry no information beyond
their components, which is why the value models leave them out (`model/domains.py`). The weights below are
**estimated from the data** (ordinary least squares over every player with ratings), not read from OOTP.
Regenerate with `PYTHONPATH=. python scripts/derived_ratings.py`.

## Short version

| Derived rating | Formula (approximate) | Check |
|---|---|---|
| Batting **CON** (contact) | **CON ≈ 0.5 × BABIP + 0.5 × AVK** (`K's`) | R² 0.97; residual SD about 1.5 rating points |
| Pitching **MOV** (movement) | **MOV ≈ 0.8 × HRR + 0.2 × PBABIP** (HRR = home runs allowed, `HRA`) | R² 0.93; residual SD about 2 rating points |

- The same relations hold for the vs-right, vs-left and potential versions of each rating (potential leans a little
  more on the second component: MOV ≈ 0.7 to 0.75 HRR + 0.25 to 0.3 PBABIP).
- Every rating is a multiple of 5 (20, 25, ..., 80), so the rounding of the three ratings alone leaves a residual
  of roughly 1.5 to 1.8 points even for an exact formula: CON is as close as that allows, MOV slightly looser.
- The fitted weights differ a little between saves (MOV: 0.84 / 0.18 in Model-Sim, 0.79 / 0.23 in TGS); 80 / 20 is
  the round number inside that range.
- Practical reading: **contact is half BABIP and half strikeout avoidance; movement is mostly home-run suppression
  with a fifth from BABIP suppression.** A pitcher's value therefore already contains his `PBABIP` twice, once
  directly and once through `MOV`, if both are used.

## Fitted weights

Free fit: `derived = intercept + w1 × first + w2 × second`. 'Weights adding to 1' fixes the intercept at 0 and
the weights to sum to 1. Residual SD is in rating points for the constrained fit.

### Batting: CON from BABIP and AVK

| Version | Save | Players | Weight of BABIP | Weight of AVK | Intercept | R² | Weights adding to 1 | Residual SD |
|---|---|---:|---:|---:|---:|---:|---|---:|
| current (`CON` from `BABIP`, `K's`) | Model-Sim | 437,957 | 0.50 | 0.48 | +0.2 | 0.974 | 51% / 49% | 1.5 |
| current (`CON` from `BABIP`, `K's`) | TGS | 9,605 | 0.49 | 0.50 | -0.1 | 0.976 | 50% / 50% | 1.6 |
| vs right-handed pitching (`CON vR` from `BA vR`, `K vR`) | Model-Sim | 437,957 | 0.50 | 0.49 | +0.1 | 0.974 | 51% / 49% | 1.5 |
| vs right-handed pitching (`CON vR` from `BA vR`, `K vR`) | TGS | 9,605 | 0.49 | 0.50 | -0.1 | 0.976 | 49% / 51% | 1.6 |
| vs left-handed pitching (`CON vL` from `BA vL`, `K vL`) | Model-Sim | 437,957 | 0.50 | 0.49 | +0.1 | 0.974 | 51% / 49% | 1.5 |
| vs left-handed pitching (`CON vL` from `BA vL`, `K vL`) | TGS | 9,605 | 0.50 | 0.50 | -0.1 | 0.977 | 50% / 50% | 1.6 |
| potential (`CON P` from `BA P`, `K P`) | Model-Sim | 437,957 | 0.52 | 0.47 | +0.1 | 0.982 | 52% / 48% | 1.8 |
| potential (`CON P` from `BA P`, `K P`) | TGS | 9,605 | 0.52 | 0.49 | -0.5 | 0.981 | 52% / 48% | 1.7 |

### Pitching: MOV from HRR and PBABIP

| Version | Save | Players | Weight of HRR | Weight of PBABIP | Intercept | R² | Weights adding to 1 | Residual SD |
|---|---|---:|---:|---:|---:|---:|---|---:|
| current (`MOV` from `HRR`, `PBABIP`) | Model-Sim | 437,957 | 0.84 | 0.18 | -1.0 | 0.934 | 84% / 16% | 2.3 |
| current (`MOV` from `HRR`, `PBABIP`) | TGS | 9,605 | 0.79 | 0.23 | -1.0 | 0.974 | 81% / 19% | 1.6 |
| vs right-handed batters (`MOV vR` from `HRR vR`, `PBABIP vR`) | Model-Sim | 437,957 | 0.84 | 0.19 | -1.0 | 0.938 | 83% / 17% | 2.2 |
| vs right-handed batters (`MOV vR` from `HRR vR`, `PBABIP vR`) | TGS | 9,605 | 0.78 | 0.24 | -1.0 | 0.975 | 81% / 19% | 1.6 |
| vs left-handed batters (`MOV vL` from `HRR vL`, `PBABIP vL`) | Model-Sim | 437,957 | 0.84 | 0.18 | -1.0 | 0.935 | 84% / 16% | 2.2 |
| vs left-handed batters (`MOV vL` from `HRR vL`, `PBABIP vL`) | TGS | 9,605 | 0.78 | 0.23 | -0.9 | 0.973 | 81% / 19% | 1.6 |
| potential (`MOV P` from `HRR P`, `PBABIP P`) | Model-Sim | 437,957 | 0.73 | 0.24 | +0.9 | 0.971 | 75% / 25% | 2.0 |
| potential (`MOV P` from `HRR P`, `PBABIP P`) | TGS | 9,605 | 0.71 | 0.29 | -0.3 | 0.974 | 70% / 30% | 1.8 |
