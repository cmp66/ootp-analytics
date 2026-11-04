# Script reference

Every script takes `-h` for its full options. Run from the repository root as `PYTHONPATH=. python scripts/<name>.py ...`. A *save* is a folder under `files/` with a table in `config/saves.toml`. For the order to run things in see [workflows.md](workflows.md).

Times are rough, for the current data on the CPU.

## Everyday

### update_all.py
One-button update of the human leagues of `config/update.toml`.

```
update_all.py                      # every save, latest season
update_all.py TGS                  # one save
update_all.py --season 2046        # another season
update_all.py --only outlook lists # some steps (process, values, outlook, lists)
update_all.py --if-changed         # skip saves whose input files are unchanged
update_all.py --steps              # list the steps
```
Runs, per save: `process_seasons`, `predict_values` (with the `models_from` save), `predict_outlook`, and the lists of `config/viewer.toml` into `output/lists/`. The lists that compare with your roster need `org` in `config/update.toml`. Does not train. About 35 s for two saves.

### run_viewer.py
`run_viewer.py [--port 8501]` starts the Streamlit viewer at http://localhost:8501: organization overview (default; org selector with all organizations and free agents), minor leagues (ceiling / floor with the chance of reaching it, 5-season or career window), MLB draft, IAFA, the acquisitions page (where the MLB roster needs an upgrade and the internal, waiver and free agent candidates within the budget space; `config/acquisition.toml`), the level moves page (which minor leaguers should move up or down, from their ratings relative to the level; `config/minors.toml`), the depth chart (an OOTP-style field for one organization: entire organization, minor leagues or one level, with players over the level's age / pro-years limits of `config/saves.toml` in red, ranked by current value or the 10-year ceiling / floor) and the explorer. Click a column header to sort, the Pos header to filter positions, a player for the ratings page. Reads the csv files `update_all.py` writes.

### outlook_lists.py
Prints or saves one list of `config/viewer.toml`.
```
outlook_lists.py TGS --lists                                  # list names
outlook_lists.py TGS --list very_good --slot SS --horizon 3
outlook_lists.py TGS --list trade_targets --org CLE --slot SP --top 20
outlook_lists.py TGS --list draft --out draft.csv
```
Options: `--season`, `--org`, `--horizon {1,2,3,5,7,10,99}` (99 = career), `--slot`, `--top`, `--out`. Needs the outlook files of `predict_outlook.py`.

## Pipeline steps

### process_seasons.py
Raw OOTP files to the processed csv files (stats, run values, park adjustment, fielding tiers) in `files/<save>/<season>/output/`.
```
process_seasons.py TGS                       # latest season
process_seasons.py TGS --season 2046
process_seasons.py Model-Sim --start 2045 --stop 2050
process_seasons.py Model-Sim --list          # seasons that have data
```
A failing season is reported and the rest still run (exit status 1).

### predict_values.py
Scores the ratings of a season with the value models and writes `<save>-<season>-predicted-values.csv`.
```
predict_values.py TGS --models-from Model-Sim [--season 2046] [--out file.csv]
```
Columns: batting / baserunning (runs per 600 PA), pitching (runs per 200 IP), fielding by position (runs per season), plus league-relative `_rel` twins. The ratings set of the models must match the save's.

### predict_outlook.py
Scores every player with the outlook models and writes `<save>-<season>-outlook-slots.csv` (one row per player and slot) and `-outlook-players.csv` (one row per player).
```
predict_outlook.py TGS --models-from Model-Sim [--season 2046] [--depth-percentile 0.3]
```
Horizons 1, 5, 7, 10 seasons ahead (the models of 2 and 3 were dropped; older output files may still have them) and the career view (`_h99`). Columns: [outlook-columns.md](outlook-columns.md). About 15 s.

## Training

### train_models.py
Trains the value models of a save into `files/models/<save>/<ratings>/` (one `.joblib` per model).
```
train_models.py Model-Sim --start 2046 --stop 2086
train_models.py Model-Sim --streams batting baserunning --no-cv
```
`--streams park` trains the four neutral-park event-rate models of the park adjustment (home runs, doubles, triples, hits per plate appearance, from the hitters' ratings; `model/park.py`) into the `park` folder of the model set; the default (all streams) trains them too. For a league whose ratings are not all 20-80 add `--scales <league>`: the training ratings are turned into the mix of scales that league shows (its 1-10 columns, converted with `config/rating-scales/<version>.csv`) and the models go to `files/models/<save>/<ratings>@<league>/`. The league names that set in the `scales` key of `config/saves.toml`. Takes minutes.

Options: `--streams` (batting, baserunning, pitching-fip, pitching-babip, fielding), `--no-plain` (skip the total fallbacks), `--no-cv` (skip the honest scores; faster), `--device cpu|cuda|xpu` (or `OOTP_DEVICE`), `--seed`. Needs processed output for the seasons.

### build_outlook_tables.py
Builds the longitudinal tables (`players.csv.gz`, `values.csv.gz`) in `files/<save>/outlook/` from the processed output and the value models. About 2 minutes.
```
build_outlook_tables.py Model-Sim [--start 2046 --stop 2060] [--models-from Model-Sim] [--remeasure-gates]
```
`--remeasure-gates` measures the position minimums into `config/outlook.toml` again (done automatically the first time for a ratings set).

### train_outlook_models.py
Trains the outlook models into `files/models/<save>/<ratings>/outlook/` and writes `docs/outlook-model-evaluation-<save>.md`. About 1 hour per game version.
```
train_outlook_models.py Model-Sim
train_outlook_models.py Model-Sim --horizons 99 --out docs/outlook-model-evaluation-Model-Sim-career.md
train_outlook_models.py Model-Sim --families hitters --no-eval
```
Options: `--families {hitters,pitchers}`, `--horizons` (default 1 5 7 10 99; 99 = career view), `--max-rows`, `--no-eval`, `--out`. Needs the tables of `build_outlook_tables.py`.

### sync_models.py
Stores the trained models in the S3 bucket `ootp-models` and fetches them on another machine. Uses `boto3` and the credentials of `aws configure` (the AWS CLI itself is not needed; `OOTP_MODELS_BUCKET` or `--bucket` changes the bucket).
```
sync_models.py push                  # upload the changed models, add a dated set
sync_models.py pull                  # download the latest set (only files that differ)
sync_models.py pull --version 20261103T141256Z-566a874   # an earlier set
sync_models.py list                  # the stored sets, newest first
sync_models.py push --dry-run        # say what would be transferred (also for pull)
```
Syncs every `.joblib` under `files/models/<save>/` (not the legacy `.pt` files). Each file is stored once under `blobs/<sha256>`; each push writes `manifests/<time>-<git commit>.json` listing the files of that set, so unchanged files are never uploaded twice and any earlier set can be pulled. Nothing is deleted from the bucket. A first full push is about 228 MB.

## Checks and reports

### evaluate_models.py
Scores trained value models against real outcomes.
```
evaluate_models.py Model-Sim --calibration
evaluate_models.py Model-Sim --start 2081 --stop 2086      # seasons the models never saw (train with --stop 2080)
evaluate_models.py Dugout26 --models-from Model-Sim-26-Std
```
Prints the scores stored at training (honest) and the scores on the seasons asked for (marked in-sample when trained on).

### review_outlook.py
`review_outlook.py <save> [--out file.md] [--csv-dir dir]` writes `docs/outlook-diagnostics-<save>.md`: attribute development by age, what traits and proneness change, who stays in the league, aging of value, baselines the outlook models must beat. About a minute; needs the outlook tables.

### check_format_parity.py
`check_format_parity.py [--out docs/export-dump-parity.md]` compares dump and export saves of the same ratings set: columns the models read, trait conversions, level, draft route. Run after changing the import or `model/formats.py`.

### predict_park.py
`predict_park.py <save> [--season N] [--models-from <training save>]` writes `<save>-<season>-park-rates.csv`: for every player with ratings his neutral-park rate per plate appearance of home runs, doubles, triples and hits and the share of his plate appearances from the left side. The viewer's park-adjusted values read it (`viewer/parks.py`). `update_all.py` runs it as the `park` step (skipped, with a note, when the model set has no park models).

### build_scale_tables.py
`build_scale_tables.py [--root files/Ratings-Analysis] [--versions OOTP26 OOTP27]` builds `config/rating-scales/<version>.csv` from two dumps of one league taken at the same moment, one with 20-80 ratings (`<root>/<version>/20-80/`) and one with 1-10 (`<root>/<version>/1-10/`): how often each 20-80 value is each 1-10 value, per rating column. The tables are in git; rerun only for a new game version.

### compare_scale_approaches.py
`compare_scale_approaches.py <training save> [--like <league>] [--streams ...] [--family ridge]` compares, for every model, the 20-80 reference with (A) a model retrained on ratings turned to 1-10 and (B) the 20-80 model applied to 1-10 ratings converted back, on the last seasons as test. Writes `docs/rating-scale-comparison-<save>[-<league>].md`. A (retraining) was better; that is why `train_models.py --scales` exists.

### detect_rating_scales.py
`detect_rating_scales.py <save> [--season N] [--all]` shows, for each rating the value models read, the scale of the league (20-80 or 1-10), the range, the number of different values and the step between them; `--all` covers every numeric column. Use it on a league whose ratings are not all 20-80 to see which columns are which (reads the processed player-data; `model/scales.py`).

### rating_importance.py, compare_versions.py, derived_ratings.py
Documentation generators for `docs/`:
- `rating_importance.py [save] [--out ...] [--observations ...] [--folds 5 --repeats 3]`: which ratings matter most per model (`docs/model-attribute-ranking*.md`).
- `compare_versions.py [--old Model-Sim-26-Std --new Model-Sim ...]`: OOTP 26 vs 27 attribute impact; about 5 minutes.
- `derived_ratings.py [saves] [--out docs/derived-ratings.md]`: how CON and MOV are made of their components.

## Analysis and development only

Not part of any workflow; each has its purpose in its docstring (and `plans/`, which is not in git).

| Script | Purpose |
| --- | --- |
| `baseline_models.py`, `compare_models.py`, `review_attributes.py` | Model family and attribute-list studies behind `model/domains.py`. |
| `review_metrics.py`, `validate_ability_targets.py`, `validate_fielding_ability.py`, `fielding_attribute_report.py` | Checks of the calculated metrics and fielding values. |
| `analyze_biz_babip.py`, `analyze_biz_babip_models.py`, `analyze_biz_fielders.py`, `analyze_biz_league.py`, `analyze_biz_tgs.py` | The study of whether pitcher / batter BABIP ratings move BIZ chances (no measurable effect). |
| `compare_outputs.py <dir_a> <dir_b>` | Which columns differ between two output folders (regression check after changing the processing). |
| `profile_pipeline.py` | Time per pipeline stage. |
| `make_test_data.py` | Rebuild the small fixtures in `tests/data` from the full files. |
