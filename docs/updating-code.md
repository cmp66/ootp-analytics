# Updating the code

How to change the project safely, and how a change gets to the viewer on the server. For the data side see [updating-data.md](updating-data.md).

## The loop for any change

1. **Branch** from `main` (`git switch -c <name>`). Commit in small steps; the PR is merged on GitHub.
2. **Change the code**, with a test next to it (`tests/test_<area>.py`). Fixtures are in `tests/data` (small, in git), so tests run on a fresh clone without `files/`.
3. **Format and lint:** `black viewer model scripts stats util tests` (line length 120, `pyproject.toml`) and `flake8 --max-line-length 130 viewer model scripts tests` (`setup.cfg`).
4. **Test:** `python -m pytest` (about 3 minutes; run a slice while working, e.g. `pytest tests/test_acquire.py -q`, or `-k acquisition`). Pandas performance warnings are errors on purpose (`pyproject.toml`): a fragmented frame fails the test.
5. **Look at it:** for a viewer change, `python scripts/run_viewer.py` and reload.
6. **Merge**, then update the machines that run the code (below).

Do not run the training scripts and the test suite at the same time: both use the CPU fully and a test run alongside training takes very long.

## Python environment and requirements

- Python 3.14 (pyenv virtualenv `ootp-analytics`); install the requirements file of your hardware (README, Setup).
- To add or update a library: edit `requirements.in` (shared) or `requirements-<hw>.in`, then recompile, e.g. `uv pip compile requirements-cpu.in -o requirements-cpu.txt --index-strategy unsafe-best-match`.
- **The viewer container has its own, smaller list** (`deploy/requirements-viewer.in`: Streamlit, pandas, numpy, scipy, xlsxwriter, boto3; no training libraries). If viewer code needs a new library, add it there and compile with `uv pip compile deploy/requirements-viewer.in --python 3.14 -o deploy/requirements-viewer.txt`.
- **Keep the viewer light.** Anything the viewer imports must not pull in torch, scikit-learn or shap at import: put such code behind a function-level import (see how `model/__init__.py` loads the trainers lazily; `util/update_config.py` and `model/quantiles.py` exist so the viewer does not import `scripts.update_all` or the legacy trainers). `tests/test_viewer_memory.py` fails when it happens.

## Common changes

**A new or changed rule of a screen.** The thresholds live in `config/*.toml` (see [configuration.md](configuration.md)); add the field to the `Settings` dataclass next to the loader (`viewer/acquire.py`, `viewer/rule5.py`, `viewer/minors.py`), the default to the toml with a comment, a slider on the page if it should be changeable for the session, and a test that the setting changes the result. The loaders reject unknown keys, so a typo in a toml fails loudly.

**A new viewer screen.**
1. Put the logic in a module of `viewer/` as pure functions over data frames (no Streamlit inside), the page in `viewer/screens/<name>.py`, and register it in `viewer/app.py` (and the link row in `viewer/screens/org_view.py` if it belongs in the navigation).
2. Load data through `viewer/data_cache.py` (`outlook`, `finances`, `player_data`), never with your own cache: those are shared across screens, kept as 32-bit floats and limited to two leagues. Do not change the shared frame in place; work on a copy or a selection.
3. Tests: logic in `tests/test_<name>.py`, the page in `tests/test_viewer_app.py` (Streamlit's `AppTest`), and `tests/test_viewer_memory.py` opens every page in a fresh process and checks the memory budget (500 MB) and that no training library is loaded. Fix it before merging; it protects the server.
4. Document the screen in [viewer.md](viewer.md).

**A new output of the update.** Write it in a pipeline script, add a `Step` to `STEPS` in `scripts/update_all.py` so that the update stays one command, and a line in [updating-data.md](updating-data.md) if the viewer needs it on the server (add the output name to `VIEWER_OUTPUTS` in `scripts/sync_files.py`, otherwise the server will not get it).

**A change to the processing** (`stats/`, `util/importCSV.py`, `scripts/process_seasons.py`). Reprocess a save, compare with `compare_outputs.py <dir_a> <dir_b>` to see which columns changed, and remember that the models were trained on the old processed output: if the training data changes, retrain (workflow 3 in [workflows.md](workflows.md)). Then `update_all.py`.

**A new script.** Give it a module docstring with usage, add it to [scripts.md](scripts.md) and to the one-line table in [workflows.md](workflows.md).

## Getting code to the viewer server

The viewer runs from code copied to `/opt/ootp-viewer` on the server; the data comes separately from S3.

```
VIEWER_HOST=ubuntu@<ip> deploy/deploy.sh
```

`deploy.sh` copies `config model stats util viewer scripts deploy` with rsync (never `deploy/.env`, the server's keys) and rebuilds the container, which recreates it and so restarts the viewer. On the first run it creates `deploy/.env` on the server from `deploy/.env.example` and stops so you can put the keys in; run it again. The image build installs only `deploy/requirements-viewer.txt`. Nothing outside `/opt/ootp-viewer` and the viewer container is touched.

A config-only change (a threshold in a toml) still goes through `deploy.sh`: the config is part of the image. A data-only change goes through `push_data.sh` instead.

After a deploy, look at `docker stats` and `free -m` on the server once: the viewer should stay near 350–450 MB (the container is capped at 600 MB).

## Documentation

- Prose docs are written by hand: [workflows.md](workflows.md) (jobs), [updating-data.md](updating-data.md), this file, [viewer.md](viewer.md), [scripts.md](scripts.md) (reference), [configuration.md](configuration.md), [utilities.md](utilities.md), [processing.md](processing.md), [models.md](models.md).
- Reports in `docs/` (`outlook-model-evaluation-*`, `outlook-diagnostics-*`, `model-attribute-*`, `rating-scale-comparison-*`, `export-dump-parity`, `derived-ratings`, `ootp26-vs-ootp27-attribute-impact`) are generated by scripts; regenerate them, do not edit them. The generator of each is listed in [models.md](models.md) and [scripts.md](scripts.md).
- `plans/` holds the history of the larger pieces of work; it is not in git.
