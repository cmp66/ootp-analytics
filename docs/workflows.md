# Overview and common workflows

What the project does, what each script is for in one line, and the steps for the jobs you do repeatedly. The options of every script are in [scripts.md](scripts.md). Run everything from the repository root with `PYTHONPATH=.` (and the `ootp-analytics` virtualenv active).

## How the pieces fit

```
raw OOTP files ──process_seasons──▶ processed csv (files/<save>/<season>/output/)
                                          │
                       train_models ◀─────┤   value models (files/models/<save>/<ratings>/*.joblib)
                                          │
                  predict_values ◀── models ─▶ predicted values per player
                                          │
             build_outlook_tables ◀───────┤   development tables (files/<save>/outlook/)
             train_outlook_models ◀───────┘   outlook models (files/models/<save>/<ratings>/outlook/)
                                          │
              predict_outlook ◀── outlook models ─▶ *-outlook-slots.csv, *-outlook-players.csv
                                          │
                       outlook_lists / viewer ◀── read the outlook csv files
```

- **Training leagues** (`Model-Sim`, `Model-Sim-26-Std`) are long computer-only leagues exported as database dumps. The models are trained on them, one set per game version (the *ratings set*: `Standard27` = OOTP 27, `Standard26` = OOTP 26).
- **Human leagues** (`TGS` = OOTP 27, `TGS26` = OOTP 26) are in-game exports. They are never trained on; they are scored with the models of the same game version (`models_from` in `config/update.toml`).
- **Trained models are in S3** (bucket `ootp-models`, pushed and pulled with `sync_models.py`), so a new machine gets them with one command instead of training. Nothing under `files/` is in git.

## Scripts in one line each

**Everyday**

| Script | What it does |
| --- | --- |
| `update_all.py` | One button: process, value, outlook-score and list every human league after new ratings files. |
| `sync_models.py` | Pushes the trained models to the S3 bucket (with history) or pulls them onto another machine. |
| `run_viewer.py` | Starts the viewer (org overview, minor leagues, draft, IAFA, explorer) at http://localhost:8501. |
| `outlook_lists.py` | Prints or saves one list of `config/viewer.toml` (very good, good, ranking, draft, IAFA, free agents, trade targets, ...). |

**The pipeline steps** (what `update_all.py` calls; run them alone for a training league or one step)

| Script | What it does |
| --- | --- |
| `process_seasons.py` | Raw OOTP files to processed csv files (stats, values, park adjustment). |
| `predict_values.py` | Scores players' ratings with the value models: batting, baserunning, pitching, fielding. |
| `predict_outlook.py` | Scores players with the outlook models: floor/median/ceiling, tier chances, staying, availability, injury cost. |

**Training**

| Script | What it does |
| --- | --- |
| `train_models.py` | Trains the value models of a save. |
| `build_outlook_tables.py` | Builds the longitudinal tables the outlook models learn from (needs the value models). |
| `train_outlook_models.py` | Trains the outlook models and writes the evaluation report (about an hour per game version). |

**Checks and reports** (after training or after a game update)

| Script | What it does |
| --- | --- |
| `evaluate_models.py` | Scores trained models against real outcomes of some seasons. |
| `review_outlook.py` | Writes `docs/outlook-diagnostics-<save>.md`: how ratings develop, who stays, what traits change. |
| `check_format_parity.py` | Writes `docs/export-dump-parity.md`: what the dump and the export have in common. |
| `rating_importance.py`, `compare_versions.py`, `derived_ratings.py` | Attribute rankings and OOTP 26 vs 27 comparison docs. |

**Analysis and development only** (not part of any workflow): `baseline_models.py`, `compare_models.py`, `review_attributes.py`, `review_metrics.py`, `validate_ability_targets.py`, `validate_fielding_ability.py`, `fielding_attribute_report.py`, `analyze_biz_*.py`, `compare_outputs.py`, `profile_pipeline.py`, `make_test_data.py`.

## Workflows

### 1. I exported new ratings (or statistics) files for a human league

This is the usual case. Cost: about 35 seconds for two saves.

1. Put the export in the save's folder: `files/TGS/<season>/` with the ratings in `files/TGS/<season>/ratings/` (same layout as the earlier seasons).
2. `PYTHONPATH=. python scripts/update_all.py` (all saves of `config/update.toml`) or `update_all.py TGS` (one save). Add `--season 2046` if it is not the latest season in the folder, `--if-changed` to skip a save whose files have not changed.
3. `PYTHONPATH=. python scripts/run_viewer.py` and open http://localhost:8501. Reload the page if it was already open.
4. The csv lists are also in `files/<save>/<season>/output/lists/`.

If a step fails, the summary at the end of the run names it and the exit status is 1; the other saves still run. To redo one step: `update_all.py TGS --only outlook lists`. The step names: `update_all.py --steps`.

### 2. I start a new human league (or a new season file naming)

1. Add a table to `config/saves.toml` (`source = "export"`, `ratings = "Standard27"` or `"Standard26"`, optional `ballparks`, and a `[<save>.team_names]` table mapping the team abbreviations to the names in the ballpark csv).
2. Add the save to `config/update.toml` with `models_from` set to the training league of the same game version, and `org = "<your team abbreviation>"`. The first save of that file is the one the viewer opens.
3. Put the files in `files/<save>/<season>/` and run workflow 1.

### 3. I want to refresh the models

Two kinds. The **value models** turn ratings into runs. The **outlook models** project those values forward and depend on the value models, so retrain them after the value models change.

*Retrain the value models* (new training seasons, changed attribute lists, a fix to the processing):

1. `python scripts/process_seasons.py Model-Sim --start 2046` (reprocess the training seasons if the processing changed).
2. `python scripts/train_models.py Model-Sim --start 2046 --stop 2086`; add `--no-cv` to skip the honest scores (faster).
3. `python scripts/evaluate_models.py Model-Sim --calibration` to check the scores.
4. Continue with the outlook models below, because their value tables come from these models.

*Retrain the outlook models* (about an hour per game version):

1. `python scripts/build_outlook_tables.py Model-Sim` (about 2 minutes). Add `--remeasure-gates` only if the position minimums in `config/outlook.toml` should be measured again.
2. `python scripts/train_outlook_models.py Model-Sim` (the report goes to `docs/outlook-model-evaluation-Model-Sim.md`). `--horizons 99` trains only the career view, `--no-eval` skips the evaluation, `--families hitters` limits it.
3. Repeat for `Model-Sim-26-Std` if the OOTP 26 league is affected.
4. `python scripts/update_all.py` so the human leagues use the new models.
5. `python -m pytest`, then `python scripts/sync_models.py push` to store the new set (only the changed files are uploaded; every earlier set stays available, `sync_models.py list` shows them). Push when the models are ones you want to keep, not after experiments.

Run the training scripts with nothing else heavy going on: they use the CPU fully and a test run alongside them takes very long.

### 4. A new OOTP version (or a new training league)

1. Run a computer-only league and export the database dumps to `files/<save>/dumps/<ratings>/<season>/`.
2. Add the save to `config/saves.toml` (`source = "dump"`, a new `ratings` name).
3. `process_seasons.py <save> --start <first>`, then `train_models.py <save>`.
4. `build_outlook_tables.py <save>` (this measures the position minimums into `config/outlook.toml` the first time), then `review_outlook.py <save>` and `train_outlook_models.py <save>`.
5. `check_format_parity.py` to see whether the export and the dump still agree, and adjust `model/formats.py` if columns changed.
6. Point a human league's `models_from` at the new save in `config/update.toml`.

### 5. I changed the lists or the viewer

- The age and pro-years limits of each minor league level are in `config/saves.toml` (`[TGS.levels."AA"]`, `max_age`, `max_pro_years`; 0 = no limit). The depth chart marks players over them; reload the page after editing.

- Lists are defined in `config/viewer.toml`. Edit it and run `update_all.py --only lists`, or look at one with `outlook_lists.py TGS --list <name>`.
- The viewer reads the csv files; code changes show after a reload (Streamlit offers "Rerun"). A new output should become a step in `STEPS` of `scripts/update_all.py` so that it stays the one command.

### 6. Setting up another machine

1. `git clone` the repository and check out the branch.
2. Create the Python 3.14 environment and install the requirements file of the hardware (see the README, Setup), and run `aws configure` with the credentials that may use the bucket.
3. `python scripts/sync_models.py pull` downloads the latest model set into `files/models/` (`--version <name>` for an earlier one). The models are pickles written by scikit-learn, PyTorch and friends: use the versions of the requirements file, or an artifact may fail to load. If one does, retrain (workflow 3).
4. Copy the raw files you need from the old machine: `files/<save>/` for the human leagues (and the training leagues only if you will retrain or process them). `files/` except the models is not in git.
5. `update_all.py`, then `run_viewer.py`. `python -m pytest` runs without the raw files.

### 7. Checking that nothing broke

`python -m pytest` (about 3 minutes) and `flake8 --max-line-length 130 viewer model scripts tests`. After retraining, also read the evaluation reports in `docs/outlook-model-evaluation-*.md` and compare with the previous ones in git.

## Where things are written

| What | Where |
| --- | --- |
| Raw files | `files/<save>/...` (not in git) |
| Processed csv, predicted values, outlook scores, lists | `files/<save>/<season>/output/` (not in git) |
| Outlook training tables | `files/<save>/outlook/` (not in git) |
| Value models | `files/models/<save>/<ratings>/*.joblib` (not in git; in S3 `ootp-models`) |
| Outlook models | `files/models/<save>/<ratings>/outlook/*.joblib` (not in git; in S3) |
| Evaluation and diagnostics reports | `docs/` |
| Settings | `config/saves.toml`, `config/update.toml`, `config/viewer.toml`, `config/outlook.toml` |
