# Utilities and shared code

The small modules the scripts and the viewer share, and the shell scripts of the server. The scripts themselves are in [scripts.md](scripts.md), the settings in [configuration.md](configuration.md).

## util/ (shared helpers)

| Module | What it is for |
| --- | --- |
| `util/config.py` | Reads `config/saves.toml` into `SaveConfig` objects: `load_saves()`, `get_save(name)`; `PROJECT_ROOT`. `LevelLimits` holds the age and pro-years limits of a minor league level (the order of the tables is the promotion ladder). |
| `util/update_config.py` | Reads `config/update.toml` (the human leagues, their `models_from`, `org`, `steps`): `load_update_config()`. Light on purpose: the viewer imports it instead of `scripts/update_all.py`, which would pull in the training libraries. |
| `util/paths.py` | Where the files of a save are: `save_dir`, `export_dir` (season folder), `export_ratings_dir`, `finance_dir`, `dump_dir`, `output_dir`, `output_file(save, season, "outlook-slots")` -> `files/<save>/<season>/output/<save>-<season>-outlook-slots.csv`. Use these, never build a path by hand. |
| `util/seasons.py` | `available_seasons(save)` (seasons with raw data), `select_seasons(...)` (latest by default, `--season`, `--start/--stop`), `served_seasons(save)` (seasons the viewer can show: raw data **or** processed outlook files, so a server that only has the outputs works). |
| `util/importCSV.py` | Reads the raw files into frames: the in-game export (ratings lists, team statistics, tells the files apart by marker columns) and the database dumps. `import_files` is what the processing calls. |
| `util/frames.py` | Keeps pandas 3 frames from fragmenting (`consolidate`, `add_constant_columns`, `numeric_columns`); a fragmentation warning is a test failure here. |
| `util/innings.py` | Outs <-> baseball innings notation (100.1 = 100 and 1/3) <-> decimal innings. |
| `util/device.py` | The torch device of the training scripts: CUDA, then XPU, then CPU; `OOTP_DEVICE` (cpu, cuda or xpu) overrides. |

## Other packages

| Package | Contents |
| --- | --- |
| `stats/` | The processing of a season: `pipeline.py` (what `process_seasons.py` runs), `batting.py`, `pitching.py`, `fielding.py`, `parks.py` (park adjustment), `woba.py` (league weights), `finances.py` (the finance export, `parse_money`). Described in [processing.md](processing.md). |
| `model/` | The value and outlook models: `train.py`, `evaluate.py`, `registry.py` / `artifact.py` (the stored model sets), `domains.py` (attribute lists and model family per stream), `formats.py` (dump vs export traits), `scales.py` (1-10 leagues), `outlook_*.py`, `park.py`, `relative.py`, `quantiles.py`. See [models.md](models.md). `model/__init__.py` loads the trainers only on first use so that importing it is light. `BSRModel.py`, `HittingModel.py`, `PitchingModel.py`, `FieldingModel.py`, `DevModels.py` and `legacy.py` are the old code. |
| `viewer/` | The screens (`viewer/screens/`) and their logic: `acquire.py`, `rule5.py`, `minors.py`, `depth.py`, `queries.py`, `parks.py`, `league.py`; `data_cache.py` is the one place the data is loaded (shared, 32-bit, two leagues at most). See [viewer.md](viewer.md). |
| `reports/` | The Excel reports of the explorer and the old `PredictionMaker.py`. |
| `tests/` | `python -m pytest`; fixtures in `tests/data`; `helpers.py` shared builders. See [updating-code.md](updating-code.md). |

The Jupyter notebooks in the repository root (`dataImport.ipynb`, `hittingmodel.ipynb`, ...) are the old workflow, kept for reference; the scripts replace them.

## deploy/ (the viewer on a server)

| File | What it does | Where it runs |
| --- | --- | --- |
| `deploy/deploy.sh [user@host]` | Copies the code to `/opt/ootp-viewer` (rsync, never `deploy/.env`) and rebuilds the container. First run creates `deploy/.env` on the server and stops. Host default: `$VIEWER_HOST`; key: `$SSH_KEY`. | Your machine |
| `deploy/push_data.sh [user@host]` | `sync_files.py push --set viewer`, then runs `refresh.sh` on the server. `$PYTHON` names the interpreter. | Your machine |
| `deploy/refresh.sh` | Pulls the viewer files from S3 inside the container and restarts the viewer only when something changed. Also usable from cron. | The server |
| `deploy/Dockerfile`, `deploy/docker-compose.viewer.yml` | The image (viewer requirements only) and the service (600 MB cap, bound to 127.0.0.1:8501, the `files` folder as a volume). | The server |
| `deploy/nginx/ootp.conf` | The nginx site (basic auth, websocket headers, https through certbot). | The server |
| `deploy/requirements-viewer.in` / `.txt` | The viewer's libraries. | Build |
| `deploy/.env.example` | The read-only AWS keys of the server (copy to `deploy/.env`, `chmod 600`). | The server |

Setup and S3 access: [deploy/README.md](../deploy/README.md). Which of these to use for what: [updating-data.md](updating-data.md) and [updating-code.md](updating-code.md).
