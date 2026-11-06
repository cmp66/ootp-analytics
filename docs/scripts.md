# Script reference

Every script takes `-h` for its full options. Run from the repository root as `PYTHONPATH=. python scripts/<name>.py ...`. A *save* is a folder under `files/` with a table in `config/saves.toml`. For the order to run things in see [workflows.md](workflows.md); for updating data or code, [updating-data.md](updating-data.md) and [updating-code.md](updating-code.md).

Times are rough, for the current data on the CPU.

## Everyday

### update_all.py
One-button update of the human leagues of `config/update.toml`.

```
update_all.py                      # every save, latest season
update_all.py TGS27                  # one save
update_all.py --season 2046        # another season
update_all.py --only outlook lists # some steps (process, values, outlook, lists)
update_all.py --if-changed         # skip saves whose input files are unchanged
update_all.py --steps              # list the steps
```
Runs, per save: `process_seasons`, `predict_values` (with the `models_from` save), `predict_outlook`, and the lists of `config/viewer.toml` into `output/lists/`. The lists that compare with your roster need `org` in `config/update.toml`. Does not train. About 35 s for two saves.

### run_viewer.py
`run_viewer.py [--port 8501]` starts the Streamlit viewer at http://localhost:8501. It reads the csv files `update_all.py` writes. The screens and what they show: [viewer.md](viewer.md).

### outlook_lists.py
Prints or saves one list of `config/viewer.toml`.
```
outlook_lists.py TGS27 --lists                                  # list names
outlook_lists.py TGS27 --list very_good --slot SS --horizon 3
outlook_lists.py TGS27 --list trade_targets --org CLE --slot SP --top 20
outlook_lists.py TGS27 --list draft --out draft.csv
```
Options: `--season`, `--org`, `--horizon {1,2,3,5,7,10,99}` (99 = career), `--slot`, `--top`, `--out`. Needs the outlook files of `predict_outlook.py`.

### sync_notes.py
Keeps what is not in git in step between your machines (workflow 9 of [workflows.md](workflows.md)): Claude Code's memory notes of the project (`~/.claude/projects/<project path>/memory/`), the `plans/` folder and, with `--set sessions`, the Claude Code conversations. Same design as `sync_models.py` (files stored once by hash, a dated manifest per push, a pull only downloads what differs), under `notes-<set>/` and `notes-blobs/` in the bucket, apart from the models and league files.
```
sync_notes.py push                  # memory notes and plans/ (when you finish some work)
sync_notes.py pull                  # on the other machine (before you start)
sync_notes.py push --all            # also the conversations; or --set sessions alone
sync_notes.py pull --dry-run        # say what would change
sync_notes.py pull --force          # also overwrite local files changed after the last push
sync_notes.py pull --mirror         # also remove local files that are not in the set
sync_notes.py list                  # the stored sets
```
A pull never deletes (unless `--mirror`) and leaves a local file alone that differs from the stored one and is newer than the set, so work done on this machine after the last push is not overwritten. The Claude Code folder of a project is named after its path, so the sets land in the right folder on a machine with another path.

### copy_exports.py
Copies the files of a new OOTP export into the latest season folder of each save of `config/update.toml` (or the saves named).
```
copy_exports.py                 # TGS27, Dugout26, PBL27
copy_exports.py Dugout26        # one save
copy_exports.py --dry-run       # say what would be copied
```
`files/exports/<save>` is a symlink to the `import_export` folder of the save in the game folder (the path differs per machine; not in git). The files wanted are the ones already in `files/<save>/<season>/ratings/` and `finance/`; other files in the export folder (extra lists, `.ootp`, park info) are ignored, and a new list has to be put in the season folder once by hand before it is picked up.
- **ratings:** the files are one export, so all of them must be newer than the copy (modified time) or none. All newer: copied. None newer: nothing to do. A mix, a file missing from the export folder, or a file without `ID` or with `B` or `T` twice (the view shows both the text and the code version): nothing is copied for that save, the problems are listed and the exit status is 1.
- **finance:** exported now and then, so a file that is not newer (or not in the export folder) is skipped without a problem. It is also left alone when the ratings files have problems.
Copies keep the modified time of the export, so running it twice copies nothing the second time. It does not run the update: follow it with `update_all.py`.

### setup_machine.py
Sets up another machine to develop and run the project (workflow 6 of [workflows.md](workflows.md)): `setup_machine.py --install cpu` (or `cuda`, `xpu`) installs the libraries with `uv pip install`, checks that they import and that the S3 bucket answers, runs `sync_models.py pull`, `sync_files.py pull` and `sync_notes.py pull`, then `update_all.py`. Needs `git clone`, Python 3.14, `uv`, `aws configure` and the machine's IP in the bucket policy first.
```
setup_machine.py --install cpu     # everything, with the libraries
setup_machine.py                   # the libraries are already there
setup_machine.py --test            # also run the tests at the end
setup_machine.py --training        # also the training leagues' dumps (16 GB)
setup_machine.py --dry-run         # say what it would do
```
A failing step stops the run and says what to fix; running it again continues cheaply (only what differs is downloaded).

## Pipeline steps

### process_seasons.py
Raw OOTP files to the processed csv files (stats, run values, park adjustment, fielding tiers) in `files/<save>/<season>/output/`.
```
process_seasons.py TGS27                       # latest season
process_seasons.py TGS27 --season 2046
process_seasons.py Model-Sim --start 2045 --stop 2050
process_seasons.py Model-Sim --list          # seasons that have data
```
A failing season is reported and the rest still run (exit status 1).

### predict_values.py
Scores the ratings of a season with the value models and writes `<save>-<season>-predicted-values.csv`.
```
predict_values.py TGS27 --models-from Model-Sim [--season 2046] [--out file.csv]
```
Columns: batting / baserunning (runs per 600 PA), pitching (runs per 200 IP), fielding by position (runs per season), plus league-relative `_rel` twins. The ratings set of the models must match the save's.

### predict_outlook.py
Scores every player with the outlook models and writes `<save>-<season>-outlook-slots.csv` (one row per player and slot) and `-outlook-players.csv` (one row per player).
```
predict_outlook.py TGS27 --models-from Model-Sim [--season 2046] [--depth-percentile 0.3]
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

### sync_files.py
Stores the league files in the same S3 bucket and fetches them on another machine (same design as `sync_models.py`: files stored once under `blobs/<sha256>`, a dated manifest per push, a pull only downloads what differs and never deletes).
```
sync_files.py push                    # the raw files of the human leagues (ratings, statistics, finance, ballparks): about 22 MB
sync_files.py pull                    # on another machine
sync_files.py push --set viewer       # what the viewer server needs: the processed files of the latest season (about 160 MB)
sync_files.py pull --set viewer
sync_files.py push --set training     # the dumps of the training leagues; very large (16 GB), only when you need to retrain elsewhere
sync_files.py list --set raw          # the stored sets, newest first
```
The processed `output` folders are left out of `raw` (`update_all.py` makes them again). Each set has its own history (`files-raw`, `files-viewer`, `files-training` manifests). The machine has to be allowed by the bucket policy. The viewer on a server: [deploy/README.md](../deploy/README.md).

## Server (deploy/)

Shell scripts, not Python. `$VIEWER_HOST` (`user@ip`) is the default host of the two that run from your machine; `$SSH_KEY` names the key. Setup: [deploy/README.md](../deploy/README.md).

### deploy/deploy.sh
`deploy/deploy.sh [user@host]` copies the code (`config model stats util viewer scripts deploy`) to `/opt/ootp-viewer` with rsync and rebuilds the container, which restarts the viewer. Never overwrites the server's `deploy/.env`; on the first run it creates it from `.env.example` and stops so you can add the keys. Use it for new code or a changed config file.

### deploy/push_data.sh
`deploy/push_data.sh [user@host]` runs `sync_files.py push --set viewer` (`$PYTHON` names the interpreter) and then `refresh.sh` on the server. Use it after `update_all.py` when the server should show the new data.

### deploy/refresh.sh
On the server: pulls the `viewer` set from S3 inside the container and restarts the viewer only when something was downloaded. Also fit for cron (`*/15 * * * * /opt/ootp-viewer/deploy/refresh.sh >> /var/log/ootp-refresh.log 2>&1`).

For the shared helper modules see [utilities.md](utilities.md).

## Checks and reports

### evaluate_models.py
Scores trained value models against real outcomes.
```
evaluate_models.py Model-Sim --calibration
evaluate_models.py Model-Sim --start 2081 --stop 2086      # seasons the models never saw (train with --stop 2080)
evaluate_models.py Dugout26 --models-from Model-Sim-26-Std
```
Prints the scores stored at training (honest) and the scores on the seasons asked for (marked in-sample when trained on).

### check_stuff_bias.py
`check_stuff_bias.py Model-Sim [--start 2081 --stop 2086] [--stuff-min 65 --others-max 50 --gap 10] [--models-from Model-Sim]` takes the two pitching models (`pitching-fip`: park adjusted FIP runs per 200 IP above average; `pitching-babip`: BABIP above the league's) and compares what they predicted with what happened, for groups of pitchers made from their ratings (the mean of the vs-right and vs-left ratings): *stuff only* (stuff at least `--stuff-min`, control, home run avoidance (HRR) and PBABIP at most `--others-max`), *control only* (control and home run avoidance at least `--stuff-min`, stuff at most `--others-max`), *balanced* (the four within `--gap` points) and *all*; starters and relievers apart. A group whose actual runs (or BABIP) are clearly above the prediction (`z` above 3) is one the model is too kind to. Needs a save with outcomes and its trained models: a training league (`Model-Sim`, `Model-Sim-26-Std`), whose dumps are not in S3 by default (see `sync_files.py push --set training`). It checks the value now; the ceiling models build on the same ratings, so a bias here is a bias there, but a clean result does not clear the ceiling.

### review_outlook.py
`review_outlook.py <save> [--out file.md] [--csv-dir dir]` writes `docs/outlook-diagnostics-<save>.md`: attribute development by age, what traits and proneness change, who stays in the league, aging of value, baselines the outlook models must beat. About a minute; needs the outlook tables.

### check_human_bias.py
Checks whether the models miss in the same way in the human leagues (workflow 10 of [workflows.md](workflows.md)): pitchers, hitters, baserunning and every fielding position. For each model and league it takes the miss of the model (outcome minus prediction), removes the league-wide shift and regresses it on the ratings in standard deviations of the league; the coefficient is the runs the model misses per standard deviation. The table has a column per league, a reference per training league (should be near zero), the pooled coefficient with its z, and `<<<` where all leagues agree in sign and the pooled |z| is at least 2.5.
```
check_human_bias.py                                          # every league of config/update.toml, latest season
check_human_bias.py --leagues TGS27 PBL27 --models pitching-fip batting
check_human_bias.py --flag-z 2 --reference-seasons 10
```
One season of a league is a few hundred players, so look for the sign agreeing between leagues, and run it again when a league has another season. The result that led to `pitching_control_per_sd` is in [models.md](models.md).

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
