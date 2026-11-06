# Overview and common workflows

What the project does and the steps for the jobs you do repeatedly. Run everything from the repository root with `PYTHONPATH=.` (and the `ootp-analytics` virtualenv active). The options of every script are in [scripts.md](scripts.md).

## I want to ...

| I want to ... | Read |
| --- | --- |
| Load a new export and look at it | [updating-data.md](updating-data.md); workflow 1 below |
| Show the new data on the server | [updating-data.md](updating-data.md), "Getting it to the viewer on the server" (`deploy/push_data.sh`) |
| Change code and get it to the server | [updating-code.md](updating-code.md) (`deploy/deploy.sh`) |
| Start a new league, or a league with 1-10 ratings | Workflows 2 and 2b |
| Retrain the models, or support a new game version | Workflows 3 and 4 |
| Change a threshold of a screen | [configuration.md](configuration.md); workflow 5 |
| Understand what a screen shows | [viewer.md](viewer.md) |
| Set up another machine or the server | Workflow 6 (`scripts/setup_machine.py`) and 8; [deploy/README.md](../deploy/README.md) |
| Switch between my two machines | Workflow 9 (`scripts/sync_notes.py`) |
| Check that nothing broke | Workflow 7 |
| Find a script or a helper module | [scripts.md](scripts.md), [utilities.md](utilities.md) |
| Know how the numbers are made | [processing.md](processing.md), [models.md](models.md) |

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
- **Human leagues** (`TGS27` = OOTP 27, `Dugout26` = OOTP 26, `PBL27` = OOTP 27; each with its own scale mix) are in-game exports. They are never trained on; they are scored with the models of the same game version (`models_from` in `config/update.toml`).
- **Trained models are in S3** (bucket `ootp-models`, pushed and pulled with `sync_models.py`), so a new machine gets them with one command instead of training. Nothing under `files/` is in git.

## Scripts in one line each

**Everyday**

| Script | What it does |
| --- | --- |
| `copy_exports.py` | Copies the new exports from each save's import_export folder (`files/exports/<save>`) into the season folder; stops if the ratings files are a mix of new and old. |
| `update_all.py` | One button: process, value, outlook-score and list every human league after new ratings files. |
| `sync_models.py` | Pushes the trained models to the S3 bucket (with history) or pulls them onto another machine. |
| `sync_notes.py` | Keeps Claude Code's memory notes, `plans/` and (on request) its conversations in step between machines. |
| `setup_machine.py` | Sets up another machine: libraries, models and files from S3, the update. |
| `sync_files.py` | The same for the league files (raw exports, the viewer's processed files, the training dumps). |
| `deploy/push_data.sh`, `deploy/deploy.sh` | New data to the viewer server; new code to the viewer server. |
| `run_viewer.py` | Starts the viewer (organization overview, minor leagues, draft, IAFA, depth chart, level moves, acquisitions, Rule 5, explorer) at http://localhost:8501. |
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
| `check_stuff_bias.py` | Checks whether the pitching models are too kind to pitchers with stuff and little else (or the opposite), against the real outcomes of a training league. |
| `review_outlook.py` | Writes `docs/outlook-diagnostics-<save>.md`: how ratings develop, who stays, what traits change. |
| `check_format_parity.py` | Writes `docs/export-dump-parity.md`: what the dump and the export have in common. |
| `rating_importance.py`, `compare_versions.py`, `derived_ratings.py` | Attribute rankings and OOTP 26 vs 27 comparison docs. |

**Analysis and development only** (not part of any workflow): `baseline_models.py`, `compare_models.py`, `review_attributes.py`, `review_metrics.py`, `validate_ability_targets.py`, `validate_fielding_ability.py`, `fielding_attribute_report.py`, `analyze_biz_*.py`, `compare_outputs.py`, `profile_pipeline.py`, `make_test_data.py`.

## Workflows

### 1. I exported new ratings (or statistics) files for a human league

This is the usual case. Cost: about 35 seconds for two saves.

1. Export from OOTP, then `PYTHONPATH=. python scripts/copy_exports.py` (`--dry-run` to look first). It copies the files into the latest season folder `files/<save>/<season>/ratings/` and `finance/`; see [scripts.md](scripts.md). By hand: put them there yourself, in the same layout as the earlier seasons.
2. `PYTHONPATH=. python scripts/update_all.py` (all saves of `config/update.toml`) or `update_all.py TGS27` (one save). Add `--season 2046` if it is not the latest season in the folder, `--if-changed` to skip a save whose files have not changed.
3. `PYTHONPATH=. python scripts/run_viewer.py` and open http://localhost:8501. Reload the page if it was already open.
4. The csv lists are also in `files/<save>/<season>/output/lists/`.

If a step fails, the summary at the end of the run names it and the exit status is 1; the other saves still run. To redo one step: `update_all.py TGS27 --only outlook lists`. The step names: `update_all.py --steps`.

### 2. I start a new human league (or a new season file naming)

1. Add a table to `config/saves.toml` (`source = "export"`, `ratings = "Standard27"` or `"Standard26"`, optional `ballparks`, and a `[<save>.team_names]` table mapping the team abbreviations to the names in the ballpark csv).
2. Add the save to `config/update.toml` with `models_from` set to the training league of the same game version, and `org = "<your team abbreviation>"`. The first save of that file is the one the viewer opens.
3. Put the files in `files/<save>/<season>/` and run workflow 1.

### 2b. The new league has ratings that are not all on the 20-80 scale

1. Add the league as in workflow 2 and process it: `process_seasons.py <league>`, then `detect_rating_scales.py <league>` shows which columns are 1-10.
2. Train the models for its mix of scales on the training league of the same game version: `train_models.py Model-Sim --scales <league>` (OOTP27) or `train_models.py Model-Sim-26-Std --scales <league>` (OOTP26). Minutes, not hours. The models go to `files/models/<save>/<ratings>@<league>/`; `sync_models.py push` stores them.
3. Set `scales = "<league>"` for the league in `config/saves.toml` and `models_from` in `config/update.toml` as usual, then run workflow 1. The outlook models stay the 20-80 ones: the league's 1-10 ratings are given to them as the 20-80 value expected for each.
4. A game version needs its conversion table in `config/rating-scales/` once (`build_scale_tables.py` from two dumps of one league taken at the same moment, one per scale). Both OOTP26 and OOTP27 have one.

TGS27 is all 20-80 and keeps using the 20-80 models.

### 3. I want to refresh the models

Two kinds. The **value models** turn ratings into runs. The **outlook models** project those values forward and depend on the value models, so retrain them after the value models change.

*Retrain the value models* (new training seasons, changed attribute lists, a fix to the processing; `train_models.py` also trains the park event-rate models of the viewer's park adjustment, `--streams park` alone does only those, in seconds):

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

- Screens are described in [viewer.md](viewer.md), all settings in [configuration.md](configuration.md).
- The Rule 5 page needs the `R5` (draft eligible) flag of the player list export and, for an exact 40-man, the `ON40` flag (column name in `config/rule5.toml`); re-export the player lists after enabling them, then run `update_all.py`.

- The rules of the acquisitions page (what counts as a need, the weight of a reliever, which free agents are left out) are in `config/acquisition.toml`. It needs the `finance` folder in the export for the budget space; without it the costs are not checked. The priority list at the top of the page orders the acquisitions by expected impact: runs over the contract (up to three years) against the starter and against what the organization's own prospects will be worth in one and two years (a ready prospect makes a long deal a bridge, a hole makes it worth it), times the chance a free agent signs (`[signing]`), and never takes the budget space under `min_budget` (default $10m, kept for the draft and IAFA); a free agent asking $8m or more (`multiyear_demand`) is assumed to want three years.

- The age and pro-years limits of each minor league level, and the promotion ladder (the order of the level tables, top level first), are in `config/saves.toml`; the rules of the level moves page (what counts as green, how many key ratings, how old is old for a level) in `config/minors.toml`. Check the ladder order of a new league.
- The age and pro-years limits of each minor league level are in `config/saves.toml` (`[TGS27.levels."AA"]`, `max_age`, `max_pro_years`; 0 = no limit). The depth chart marks players over them; reload the page after editing.

- Lists are defined in `config/viewer.toml`. Edit it and run `update_all.py --only lists`, or look at one with `outlook_lists.py TGS27 --list <name>`.
- The viewer reads the csv files; code changes show after a reload (Streamlit offers "Rerun"). A new output should become a step in `STEPS` of `scripts/update_all.py` so that it stays the one command.

### 6. Setting up another machine

By hand it is five things; `scripts/setup_machine.py` does the last four in order and stops at the first that fails.

1. **Before the script** (it cannot do these): `git clone` the repository and check out the branch; install Python 3.14 (pyenv virtualenv `ootp-analytics`) and `uv`; install the AWS CLI credentials with `aws configure` (the keys that may use the bucket `ootp-models`); and add the machine's public IP address to the bucket policy (the bucket answers only to allowed addresses).
2. **The script:** `python scripts/setup_machine.py --install cpu` (or `cuda`, `xpu`, see the README, Setup, for the hardware). It installs the libraries, checks that they import and that the bucket answers, pulls the trained models (`sync_models.py pull`) and the raw files of the human leagues (`sync_files.py pull`), and runs `update_all.py` so that the processed files exist. `--test` also runs the test suite, `--training` also pulls the dumps of the training leagues (16 GB, only to retrain here), `--dry-run` shows what it would do. Run it again at any time: it only downloads what differs and deletes nothing.
3. **Start the viewer:** `python scripts/run_viewer.py`.
4. **Claude Code on the new machine:** `CLAUDE.md` in the repository gives it the working rules and the domain notes. The memory notes and `plans/` come with the script (`sync_notes.py pull`); the conversations only if you ask (workflow 9).

The models are pickles written by scikit-learn, PyTorch and friends: the requirements file pins the versions, or an artifact may fail to load. If one does, retrain (workflow 3). The raw files come from the machine that has the exports: run `sync_files.py push` there first, and after each new export. `files/` is not in git. A machine that only develops can skip the training set and the models if it never scores (the tests run without the raw files).

### 7. Checking that nothing broke

`python -m pytest` (about 3 minutes) and `flake8 --max-line-length 130 viewer model scripts tests` (and `black --check`; see [updating-code.md](updating-code.md)). After retraining, also read the evaluation reports in `docs/outlook-model-evaluation-*.md` and compare with the previous ones in git.

### 8. The viewer on a server

The viewer runs in a container on a small server (about 350 MB of memory; `deploy/`). It reads only the processed files of the latest season of each human league. New data: after `update_all.py`, `deploy/push_data.sh` (pushes the `viewer` set to S3 and has the server pull it and restart). New code: `deploy/deploy.sh`. Setup, nginx and S3 access: [deploy/README.md](../deploy/README.md). `tests/test_viewer_memory.py` keeps every page inside the memory budget: a new screen should load its data through `viewer/data_cache.py`, never with its own cache, and must not change the shared frame.

### 9. Working on two machines

You finish a big item on one machine and continue on the other; everything is in `main` when you switch.

1. **Before you leave a machine:** merge your branch to `main`; `python scripts/sync_notes.py push` (Claude Code's memory notes and `plans/`); `sync_files.py push` if there are new exports, `sync_models.py push` if you trained models.
2. **On the other machine:** `git checkout main && git pull`, then `sync_notes.py pull` (and `sync_files.py pull`, `sync_models.py pull` as needed), then `update_all.py` if the exports changed.
3. **Claude Code's conversations** are only moved when you ask: `sync_notes.py push --set sessions` here and `pull --set sessions` there, then `claude --resume` shows them. They are big and hold everything that was on screen (paths, IP addresses, outputs); the other sets hold your notes only.
4. A pull never deletes, and it leaves alone a file you changed on this machine after the last push (it says so; `--force` overwrites, `--mirror` removes local files that are not in the set). The repository's `CLAUDE.md` holds what Claude Code needs to know about the project and how you work, so it is the same on both machines; the memory notes are for what it learns along the way.

The notes sets are stored in the bucket apart from the models and league files (`notes-*/`), where the read-only key of the viewer server cannot read them.

## Where things are written

| What | Where |
| --- | --- |
| Raw files | `files/<save>/...` (not in git) |
| Processed csv, predicted values, outlook scores, lists | `files/<save>/<season>/output/` (not in git) |
| Outlook training tables | `files/<save>/outlook/` (not in git) |
| Value models | `files/models/<save>/<ratings>/*.joblib` (not in git; in S3 `ootp-models`) |
| Outlook models | `files/models/<save>/<ratings>/outlook/*.joblib` (not in git; in S3) |
| Evaluation and diagnostics reports | `docs/` |
| Settings | `config/*.toml`, all described in [configuration.md](configuration.md) |
