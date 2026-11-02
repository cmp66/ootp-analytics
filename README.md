# ootp-analytics

Models for predicting current and future player performance from Out of the Park Baseball data.

## Setup

Python 3.14 (pyenv virtualenv `ootp-analytics`). Install the requirements file matching your hardware:

| Hardware | Install |
| --- | --- |
| CPU only (default) | `uv pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` |
| NVIDIA GPU (CUDA) | `uv pip install -r requirements-cuda.txt` |
| Intel GPU (XPU) | `uv pip install -r requirements-xpu.txt --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` |

The device is chosen at runtime by `util/device.py` (CUDA, then XPU, then CPU). Set `OOTP_DEVICE=cpu` (or `cuda`, `xpu`) to override.

To update, edit `requirements.in` (shared) or `requirements-<hw>.in` and recompile, e.g.:

`uv pip compile requirements-cpu.in -o requirements-cpu.txt --index-strategy unsafe-best-match`

## Tests

`python -m pytest` runs the unit tests and the integration cross-checks. The integration tests use the small csv fixtures in `tests/data` (about 4 MB: one Model-Sim dump season and one TGS export season), so they pass on a fresh clone without the large `files/` folder. To rebuild the fixtures from the full files run `python scripts/make_test_data.py`; to run the integration tests against the full files set `OOTP_TEST_DATA=full`.

## Park factors

`OFFAdj` (batters) and `WAAAdj` (pitchers) are park adjusted when a save lists a ballpark ratings csv in `config/saves.toml` (`ballparks = "ballpark_ratings.csv"`, kept in `files/<save>/`; `files/<save>/<season>/ballpark_ratings.csv` overrides it for one season). Without it they equal `OFF` and `WAA`. Columns: `Team Name, Park, PF AVG, AVG L, AVG R, PF HR, HR L, HR R, PF D, PF T, PF`, where 1.000 is neutral and the L / R columns are for left- and right-handed batters.

- `home_share` (default 0.5) is the share of a player's plate appearances / batters faced in his own park; the rest count as neutral.
- Team names in the csv are matched to the save's teams: dump saves use "city nickname" automatically; export saves need a `[<save>.team_names]` table (team abbreviation to csv name); any exception goes in the same table. Teams without a park are logged and treated as neutral.
- Batters: hits, doubles, triples and home runs are divided by the park factor (by batting side), then `wOBAAdj` and `wRAAAdj` are computed against the league mean of the adjusted wOBA, so league `wRAAAdj` sums to 0. New columns `ParkAVG`, `ParkHR`, `ParkD`, `ParkT`, `wOBAAdj`, `wRAAAdj`. `OFFAdj = BSR + wRAAAdj`.
- Pitchers: expected runs use the run factor `PF`; new columns `ParkRuns`, `ExpRA9Adj`; `WAAAdj` still sums to 0 in every usage bucket.

## Fielding

The make-rate fielding columns (`wPM*`, `wRunsPSeason`, `runsPAdj`, `runsPAdjSeason`, `PlayPercentAA`) judge a fielder against the league **inside each BIZ difficulty tier** (R, L, E, U, Z; "I", impossible, is ignored). `PlaysExpected` is what an average fielder at his position makes on his chances, `PlaysAboveExpected` is the difference and `PlayPercentExp` the expected make rate; the league tier rates are written to `<save>-<season>-fielding-tier-rates.csv`. This removes the effect of the staff and park on which kinds of balls a fielder sees (pitcher and batter BABIP ratings were checked and have no measurable effect). Errors in these columns are charged per chance.

The volume columns (`PlayMadeAA*`, `PlayAAASeason`, `runsPSeason`, `runsPNow`, `EAASeason`) count plays per inning, so they also depend on how many balls in play the staff allows (mostly its strikeout rate). `scripts/validate_fielding_ability.py` and `scripts/fielding_attribute_report.py` check how well the values isolate the fielder.

## Ability values for the models

Fielding: `FldRunsSeason` / `FldRunsAdj` (runs: tier-adjusted plays, catcher framing, outfield arm) are the fielding value without the noisy errors; `ErrRunsSeason` / `ErrRunsAdj` are the error charge on its own. Pitching: `WAA` splits into `FIPRuns` (home runs, walks and hit batters, strikeouts against the usage-bucket average, 13 / 3 / -2 weights), `BABIPRuns` (hits on balls in play against the league BABIP; mostly luck and defense, use `BIP` as the weight) and `OtherRuns` (sequencing, unearned runs, defense): `WAA x 10 = FIPRuns + BABIPRuns + OtherRuns`. `scripts/validate_ability_targets.py` reports how repeatable each value is.

## Processing a save

`python scripts/process_seasons.py <save>` reads the raw files of a save (a folder under `files/` with a table in `config/saves.toml`, which sets the source, ratings set and ballpark file) and writes the processed csv files to `files/<save>/<season>/output/`. By default it processes the latest season only; `--season N` does one season, `--start N` runs from N through the latest, `--start N --stop M` runs the range, and `--list` shows the seasons that have data. A failing season is reported and the rest still run (exit status 1). This replaces `dataImport.ipynb`.

## Models

The value models predict, from a player's ratings, his batting, baserunning, pitching and fielding value (runs per 600 PA, per 200 IP or per season). They are trained, evaluated and applied from scripts; the notebooks and the `Hitting/Pitching/Fielding/BSRModel` classes are the old code and stay until `DevModels` and `reports/PredictionMaker.py` are ported.

```
python scripts/train_models.py Model-Sim --start 2046 --stop 2086      # train every model of a save
python scripts/evaluate_models.py Model-Sim --calibration              # scores stored at training, scores on seasons
python scripts/predict_values.py Model-Sim --season 2086               # predicted values of one season to a csv
python scripts/predict_values.py TGS --models-from Model-Sim           # models trained on one save score another
```

- **Training data** is the processed output of a save (`scripts/process_seasons.py`); the ratings set in `config/saves.toml` names the model folder `files/models/<save>/<ratings>/`, one `.joblib` file per model (weights, scaler, feature names, seasons, scores, platoon mix).
- **Models:** batting, pitching FIP part and pitching BABIP part each have a stacked platoon model (`<stream>-split`: value against right- and left-handed opponents, blended with the league's platoon mix of the player's hand) and a plain total model as a fallback; baserunning; one fielding model per position (C, 1B to 3B, SS, LF, CF, RF). The attribute lists and the model family per stream are in `model/domains.py` (`REVIEWED_FEATURES`, `CHOSEN_FAMILY`); every listed rating exists in the dump and in the in-game export, so models trained on a dump save can score an export save.
- **Evaluation** is player-grouped: no player is on both sides of a split, the network stops on validation players taken from the training set only, and `--start/--stop` of `evaluate_models.py` can score seasons the models never saw (train with `--stop 2080`, evaluate 2081 to 2086).
- **Hardware:** the models are small; ridge and boosting run on the CPU and the networks train on the CPU by default (`--device cuda` or `OOTP_DEVICE=cuda` uses a GPU where available; the Intel NPU cannot train PyTorch models and is not visible from WSL). Artifacts are pickles: only load files you trained.
- **Output** of `predict_values.py`: `batting`, `batting_vs_right`, `batting_vs_left` (runs per 600 PA), `baserunning`, `pitching_fip*` and `pitching_runs` (runs per 200 IP), `pitching_babip` (BABIP above league), `fielding_<position>` (runs per season if he played there), and the `_plain` fallbacks.
- **League-relative values** (`<column>_rel`): each value column has a twin with the playing-time weighted mean of that column over the scored league and season subtracted (`model/relative.py`). The raw columns are values against the league the models were trained on. A league that picks players differently (for example human managers who play good defenders at SS, CF, 2B and 3B) has a different average, and the raw fielding values then run high for it (TGS26 shortstops +22 runs); the `_rel` columns are values against the league being scored. The script prints the shift it subtracted for each column. The shift is about zero for the league the models were trained on.

## Outlook tables (player development)
`python scripts/build_outlook_tables.py Model-Sim` builds, for a long CPU league, the tables the outlook models of `plans/16-development-models.md` are trained on, in `files/<save>/outlook/` (`players.csv.gz`, `values.csv.gz`; about 2 minutes for all seasons). `model/outlook_data.py` documents the columns.
- **players**: one row per player and season: age, level, entry route (international / high school draft / college draft / other) and entry season, work ethic, intelligence, leadership, the four prone fields, playing time at every level and `time_share` (a proxy for missed time), the gate ratings, and `stay_h1/2/3/5` (still in the league that many seasons later; NaN, never False, when that season is beyond the table).
- **values**: one row per player, season and slot (C, 1B, 2B, 3B, SS, LF, CF, RF, DH, SP, RP): the league-relative value of his ratings at that slot (from the plan 14 models, not from stats or playing time), the rank among the qualified players of the league and the tier (top 10, top 50, rest; the cutoffs scale with the number of teams: 30 teams give ranks 10 and 50).
- Traits come in two forms so that one model can score both file formats: the dump holds numbers (1-200), the in-game export only categories (L / N / H; proneness Iron Man / Durable / Normal / Fragile / Wrecked). `model/formats.py` turns the dump numbers into the export's categories (cut points estimated by matching the export's shares, stored in `config/outlook.toml`; proneness in three levels because Iron Man and Wrecked depend on history), and the export import keeps the categories in `<trait> cat` columns. `python scripts/check_format_parity.py` writes `docs/export-dump-parity.md`, the check of what is in both formats. Adaptability is only in the export. A player is ranked at a position when his key ratings reach the position's minimum, or his potential rating there reaches the potential minimum. The minimums are measured per ratings set (1st percentile of the top-50 MLB starters at the position, floored to a multiple of 5) and kept in `config/outlook.toml`, where they can be edited; `--remeasure-gates` measures them again.

`python scripts/review_outlook.py <save>` (stage 2 of plan 16, about a minute) writes `docs/outlook-diagnostics-<save>.md`: for one game version, how every rating develops with age and what work ethic, intelligence, proneness and missed playing time change (separate low and high tiers; leadership, loyalty and finances as placebos), who stays in the league, the aging of value, the baseline the outlook models must beat, the playing time lost to proneness, and what high school, college and international entrants must reach at entry. The csv tables go to `plans/data/outlook-review/`.

`python scripts/train_outlook_models.py <save>` (stage 3 of plan 16, about 25 minutes per game version) trains the outlook models of that version: quantiles of where a player's value is going (floor, median, ceiling), the chance of being a top 10 / top 50 player at a position, of staying in the league and of being available, for 1, 2, 3 and 5 seasons ahead. Artifacts go to `files/models/<save>/<ratings>/outlook/`; the held-out evaluation to `docs/outlook-model-evaluation-<save>.md`.

`python scripts/predict_outlook.py <save> [--models-from <save>]` (stage 4 of plan 16, about 15 seconds) scores every player of a season with the outlook models of the same game version and writes `<save>-<season>-outlook-slots.csv` and `-outlook-players.csv` next to the other outputs: the floor, median and ceiling of the value and the chance of being a top 10 / top 50 player at each position 1, 2, 3 and 5 seasons ahead, the chance of staying in the league, availability, the injury risk and cost, and a depth flag. It works on dump and export saves (for example `TGS26 --models-from Model-Sim-26-Std`). Columns: `docs/outlook-columns.md`.

`python scripts/outlook_lists.py <save> --list very_good --slot SS [--org ABBR]` (stage 5 of plan 16) prints the lists of `config/viewer.toml` from the outlook files: the very good players (chance of being a top-10 player at a position), the good ones (11th to 50th), the ordered ranking, the draft class, the IAFA prospects, free agents, trade targets, and how each player compares with your organization's roster (`p_start`, `p_roster`). The queries are pure functions in `viewer/queries.py`, which the viewer of plan 15 will call.

## Documentation

- `docs/derived-ratings.md`: how the derived ratings are made (contact `CON` is half `BABIP` and half `AVK`; movement `MOV` is about 80 % home runs allowed and 20 % `PBABIP`), with the fitted weights per save. Regenerate with `PYTHONPATH=. python scripts/derived_ratings.py`.
- `docs/model-attribute-ranking.md`: which ratings contribute most to each model's value, with the effect of +10 points and +1 standard deviation in runs. Regenerate after retraining with `PYTHONPATH=. python scripts/rating_importance.py`.
- `docs/model-attribute-ranking-ootp26.md`: the same ranking for the OOTP 26 models (`PYTHONPATH=. python scripts/rating_importance.py Model-Sim-26-Std --out docs/model-attribute-ranking-ootp26.md --observations docs/model-attribute-observations-Standard26.md`).
- `docs/ootp26-vs-ootp27-attribute-impact.md`: how the effect of each rating changed between the OOTP 26 and OOTP 27 models, ranked by the size of the change, and how the two model sets value the same ratings. Regenerate with `PYTHONPATH=. python scripts/compare_versions.py` (about 5 minutes).
