# Dump and export: what the models need and how it converts

Generated 2026-11-03 by `scripts/check_format_parity.py`. The long training leagues exist only as database dumps; the human leagues only as in-game exports (a dump needs commissioner rights). A model that scores both must use only what both hold.

## Summary

- Standard26: every column the models read is in both formats.
- Standard27: every column the models read is in both formats.

## Only in the dump

| Field | Export equivalent / conversion |
|---|---|
| college | Export: no college flag (HSC is empty). Convert: age at the draft >= 20 means college (checked below). |
| acquired, free_agent, retired, league_id, last_team_id | Export: Lev (level), TXN (last transaction), ORG / TM (team). Level converts to a group (below). |
| draft_year, draft_round, draft_overall_pick, picked_in_draft | Export: Draft, Round, Pick, OA Pick (0 = not drafted). Converts directly. |
| prone_leg, prone_back, prone_arm | Not in the export. Models for human leagues use only overall proneness. |
| injury_is_injured, injury_left, injury_career_ending | Not in the export player list. Not needed to score a player. |
| experience | Different from the export's MLY (MLB years). Not used as a feature. |
| personality_* | Converted to WE, INT, LEA, LOY, FIN in the player-data of both formats (numbers 1-200 in the dump, 40 / 100 / 160 codes in the export; use the `<trait> cat` categories). |

## Only in the export

| Field | Note |
|---|---|
| AD (adaptability) | In no dump file: it cannot be modelled; the viewer can show it from the export. |
| Lev, LG, TXN, Draft, Round, Pick, OA Pick, DOB, OVR, POT | Used for level and entry fields; OVR / POT are the game's star ratings. |
| MLY, MLD, YL, ROOKIE, R5, COMP, Type, PT, SLR, ... | Service time, rookie status, personality type and contract fields; not used by the models. |

## Not available in the export at all

- Adaptability in the dump (so no development effect can be measured).
- Playing time of earlier seasons (a human save has few seasons; the lagged playing-time features need two or more).
- Entry route of undrafted players: international signings cannot be told from other undrafted entries once they are not new.

## Ratings set Standard26: Model-Sim-26-Std 2099 (dump) against Dugout26 2198 (export)

Dump player-data 6507 rows, export 13919 rows.

### Rating and context columns the models read

| Column | In dump | In export | Dump min-max | Export min-max | On the 5-point grid, dump / export |
|---|---|---|---|---|---:|
| SPE | yes | yes | 20-80 | 1-10 | 100% / 7% |
| STE | yes | yes | 20-80 | 1-10 | 100% / 7% |
| RUN | yes | yes | 20-80 | 1-10 | 100% / 10% |
| BA vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| BA vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| GAP vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| GAP vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| POW vR | yes | yes | 20-80 | 20-75 | 100% / 100% |
| POW vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| EYE vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| EYE vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| K vR | yes | yes | 20-80 | 20-75 | 100% / 100% |
| K vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| B | yes | yes | 1-3 | 1-3 | 0% / 0% |
| C FRM | yes | yes | 20-75 | 1-10 | 100% / 1% |
| C ABI | yes | yes | 20-80 | 1-10 | 100% / 1% |
| IF RNG | yes | yes | 20-75 | 1-10 | 100% / 7% |
| HT | yes | yes | 173-211 | 172.72-210.82 | 20% / 0% |
| IF ARM | yes | yes | 20-80 | 1-10 | 100% / 22% |
| IF ERR | yes | yes | 20-80 | 1-10 | 100% / 6% |
| OF RNG | yes | yes | 20-80 | 1-10 | 100% / 27% |
| OF ARM | yes | yes | 20-80 | 1-10 | 100% / 28% |
| STU vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| STU vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| CON.1 vR | yes | yes | 20-80 | 20-65 | 100% / 100% |
| CON.1 vL | yes | yes | 20-80 | 20-70 | 100% / 100% |
| PBABIP vR | yes | yes | 20-80 | 20-70 | 100% / 100% |
| PBABIP vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| HRR vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| HRR vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| T | yes | yes | 1-2 | 1-2 | 0% / 0% |
| STM | yes | yes | 20-80 | 1-10 | 100% / 7% |
| PIT | yes | yes | 0-6 | 0-6 | 57% / 57% |
| C | yes | yes | 0-75 | 0-9 | 100% / 92% |
| 1B | yes | yes | 0-80 | 0-10 | 100% / 85% |
| 2B | yes | yes | 0-75 | 0-10 | 100% / 87% |
| 3B | yes | yes | 0-75 | 0-10 | 100% / 88% |
| SS | yes | yes | 0-75 | 0-10 | 100% / 89% |
| LF | yes | yes | 0-80 | 0-10 | 100% / 79% |
| CF | yes | yes | 0-80 | 0-10 | 100% / 88% |
| RF | yes | yes | 0-80 | 0-10 | 100% / 80% |
| C Pot | yes | yes | 0-75 | 0-9 | 100% / 93% |
| 1B Pot | yes | yes | 20-80 | 1-10 | 100% / 17% |
| 2B Pot | yes | yes | 0-80 | 0-10 | 100% / 79% |
| 3B Pot | yes | yes | 0-80 | 0-10 | 100% / 68% |
| SS Pot | yes | yes | 0-75 | 0-10 | 100% / 85% |
| LF Pot | yes | yes | 0-80 | 0-10 | 100% / 23% |
| CF Pot | yes | yes | 0-80 | 0-10 | 100% / 53% |
| RF Pot | yes | yes | 0-80 | 0-10 | 100% / 24% |
| WE | yes | yes | 1-200 | 40-160 | 21% / 100% |
| INT | yes | yes | 1-200 | 40-160 | 20% / 100% |
| LEA | yes | yes | 1-200 | 40-160 | 20% / 100% |
| LOY | yes | yes | 1-200 | 40-160 | 20% / 100% |
| FIN | yes | yes | 1-200 | 40-160 | 20% / 100% |
| Prone | yes | yes | 1-200 | 40-160 | 24% / 100% |
| Age | yes | yes | 16-44 | 15-41 | 19% / 17% |
| WT | yes | yes | 154-260 | 150-260 | 21% / 100% |

### Traits and proneness: numbers in the dump, categories in the export

Higher proneness is more injury prone (it goes with less playing time). Iron Man (sustained 20 or less) and Wrecked (sustained 180 or more) depend on history an export does not carry, so proneness is used in three levels (Durable = Iron Man + Durable, Normal, Fragile = Fragile + Wrecked).

| Trait | Export shares (labels low to high) | Cut points on the dump numbers | Dump mean of each category |
|---|---|---|---|
| WE | L 12.5%, N 64.2%, H 23.3% | 59, 141 | L 33, N 103, H 173 |
| INT | L 12.1%, N 75.4%, H 12.5% | 59, 140 | L 33, N 100, H 166 |
| LEA | L 15.9%, N 73.6%, H 10.5% | 57, 140 | L 30, N 99, H 162 |
| LOY | L 14.2%, N 69.6%, H 16.2% | 59, 137 | L 30, N 101, H 166 |
| FIN | L 17.6%, N 66.2%, H 16.2% | 61, 140 | L 33, N 101, H 169 |
| Prone | Durable 11.0%, Normal 80.2%, Fragile 8.8% | 34, 120 | Durable 12, Normal 75, Fragile 166 |

### Level

| Level group | Dump share | Export share |
|---|---:|---:|
| A | 31.6% | 10.7% |
| Rookie | 20.1% | 25.6% |
| MLB | 15.1% | 7.4% |
| AAA | 15.0% | 10.0% |
| AA | 14.8% | 9.3% |
| other | 3.2% | 15.2% |
| FA | 0.2% | 21.8% |

### Draft route

Drafted players of 11 sample seasons of Model-Sim-26-Std: 35711. The rule 'college from age 20' agrees with the college flag for 99.0% of them. (Rule from age 19: 85.3%.)

| Age at the draft | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Players | 1103 | 13832 | 5565 | 381 | 6299 | 8069 | 459 |
| College flag | 0.0% | 0.2% | 6.1% | 98.7% | 100.0% | 100.0% | 100.0% |

## Ratings set Standard27: Model-Sim 2086 (dump) against TGS27 2046 (export)

Dump player-data 10307 rows, export 10561 rows.

### Rating and context columns the models read

| Column | In dump | In export | Dump min-max | Export min-max | On the 5-point grid, dump / export |
|---|---|---|---|---|---:|
| SPE | yes | yes | 20-80 | 20-80 | 100% / 100% |
| STE | yes | yes | 20-80 | 20-80 | 100% / 100% |
| RUN | yes | yes | 0-80 | 0-80 | 100% / 100% |
| BA vR | yes | yes | 20-80 | 20-75 | 100% / 100% |
| BA vL | yes | yes | 20-80 | 20-75 | 100% / 100% |
| GAP vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| GAP vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| POW vR | yes | yes | 20-80 | 20-75 | 100% / 100% |
| POW vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| EYE vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| EYE vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| K vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| K vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| B | yes | yes | 1-3 | 1-3 | 0% / 0% |
| C FRM | yes | yes | 20-80 | 20-80 | 100% / 100% |
| C ABI | yes | yes | 20-80 | 20-80 | 100% / 100% |
| IF RNG | yes | yes | 20-80 | 20-80 | 100% / 100% |
| HT | yes | yes | 171-211 | 172.72-210.82 | 20% / 0% |
| IF ARM | yes | yes | 20-80 | 20-80 | 100% / 100% |
| IF ERR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| OF RNG | yes | yes | 20-75 | 20-80 | 100% / 100% |
| OF ARM | yes | yes | 20-80 | 20-80 | 100% / 100% |
| STU vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| STU vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| CON.1 vR | yes | yes | 20-75 | 0-75 | 100% / 100% |
| CON.1 vL | yes | yes | 20-75 | 0-75 | 100% / 100% |
| PBABIP vR | yes | yes | 20-65 | 20-65 | 100% / 100% |
| PBABIP vL | yes | yes | 20-70 | 20-70 | 100% / 100% |
| HRR vR | yes | yes | 20-80 | 20-80 | 100% / 100% |
| HRR vL | yes | yes | 20-80 | 20-80 | 100% / 100% |
| T | yes | yes | 1-2 | 1-2 | 0% / 0% |
| STM | yes | yes | 20-80 | 20-80 | 100% / 100% |
| PIT | yes | yes | 0-6 | 0-6 | 55% / 57% |
| C | yes | yes | 0-80 | 0-80 | 100% / 100% |
| 1B | yes | yes | 0-80 | 0-80 | 100% / 100% |
| 2B | yes | yes | 0-80 | 0-80 | 100% / 100% |
| 3B | yes | yes | 0-80 | 0-80 | 100% / 100% |
| SS | yes | yes | 0-80 | 0-80 | 100% / 100% |
| LF | yes | yes | 0-80 | 0-80 | 100% / 100% |
| CF | yes | yes | 0-75 | 0-80 | 100% / 100% |
| RF | yes | yes | 0-80 | 0-80 | 100% / 100% |
| C Pot | yes | yes | 0-80 | 0-80 | 100% / 100% |
| 1B Pot | yes | yes | 20-80 | 20-80 | 100% / 100% |
| 2B Pot | yes | yes | 0-80 | 0-80 | 100% / 100% |
| 3B Pot | yes | yes | 0-80 | 0-80 | 100% / 100% |
| SS Pot | yes | yes | 0-80 | 0-80 | 100% / 100% |
| LF Pot | yes | yes | 0-80 | 0-80 | 100% / 100% |
| CF Pot | yes | yes | 0-75 | 0-80 | 100% / 100% |
| RF Pot | yes | yes | 0-80 | 0-80 | 100% / 100% |
| WE | yes | yes | 1-200 | 40-160 | 20% / 100% |
| INT | yes | yes | 1-200 | 40-160 | 20% / 100% |
| LEA | yes | yes | 1-200 | 40-160 | 20% / 100% |
| LOY | yes | yes | 1-200 | 40-160 | 20% / 100% |
| FIN | yes | yes | 1-200 | 40-160 | 20% / 100% |
| Prone | yes | yes | 0-200 | 40-160 | 19% / 100% |
| Age | yes | yes | 15-45 | 16-41 | 17% / 19% |
| WT | yes | yes | 151-263 | 150-255 | 20% / 100% |

### Traits and proneness: numbers in the dump, categories in the export

Higher proneness is more injury prone (it goes with less playing time). Iron Man (sustained 20 or less) and Wrecked (sustained 180 or more) depend on history an export does not carry, so proneness is used in three levels (Durable = Iron Man + Durable, Normal, Fragile = Fragile + Wrecked).

| Trait | Export shares (labels low to high) | Cut points on the dump numbers | Dump mean of each category |
|---|---|---|---|
| WE | L 7.5%, N 67.3%, H 25.2% | 33, 132 | L 18, N 94, H 169 |
| INT | L 12.1%, N 74.6%, H 13.3% | 58, 131 | L 33, N 99, H 164 |
| LEA | L 15.4%, N 73.2%, H 11.4% | 60, 136 | L 34, N 97, H 160 |
| LOY | L 14.1%, N 68.3%, H 17.6% | 61, 135 | L 31, N 101, H 166 |
| FIN | L 18.2%, N 65.9%, H 15.9% | 61, 136 | L 32, N 100, H 169 |
| Prone | Durable 11.0%, Normal 72.0%, Fragile 17.0% | 30, 87 | Durable 9, Normal 61, Fragile 106 |

### Level

| Level group | Dump share | Export share |
|---|---:|---:|
| Rookie | 39.6% | 19.4% |
| A | 22.4% | 18.0% |
| AA | 11.8% | 8.2% |
| AAA | 9.4% | 8.0% |
| MLB | 8.9% | 8.4% |
| other | 7.3% | 9.6% |
| FA | 0.6% | 28.3% |

### Draft route

Drafted players of 7 sample seasons of Model-Sim: 19349. The rule 'college from age 20' agrees with the college flag for 98.8% of them. (Rule from age 19: 83.9%.)

| Age at the draft | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Players | 608 | 7922 | 3323 | 220 | 3124 | 3853 | 298 |
| College flag | 0.0% | 0.2% | 6.6% | 99.1% | 100.0% | 100.0% | 100.0% |
