# ootp-analytics

Models for predicting current and future player performance from Out of the Park Baseball data.

## Where to start

Organized by what you want to do:

- [docs/workflows.md](docs/workflows.md): "I want to ..." index and the step-by-step workflows (new export, new league, retraining, new machine, the server).
- [docs/updating-data.md](docs/updating-data.md): new exports -> `update_all.py` -> the viewer, and to the server.
- [docs/updating-code.md](docs/updating-code.md): changing the code, tests, adding a screen or a step, and deploying code to the server.
- [docs/viewer.md](docs/viewer.md): every screen and how to read it.
- [docs/configuration.md](docs/configuration.md): every `config/*.toml` and what to do after editing it.

Reference:

- [docs/scripts.md](docs/scripts.md): usage, options and outputs of every script (and the deploy scripts).
- [docs/utilities.md](docs/utilities.md): `util/`, the other packages and `deploy/`.
- [docs/processing.md](docs/processing.md): how the statistics, park factors, fielding and ability values are made.
- [docs/models.md](docs/models.md): the value and outlook models, leagues with other rating scales, generated reports.
- [deploy/README.md](deploy/README.md): the viewer on a server.

`files/` is not in git. The trained models and the league files are kept in the S3 bucket `ootp-models` with a history of every set (`sync_models.py`, `sync_files.py`; needs `aws configure` credentials).

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

`python -m pytest` runs the unit tests and the integration cross-checks. The integration tests use the small csv fixtures in `tests/data` (about 4 MB: one Model-Sim dump season and one TGS27 export season), so they pass on a fresh clone without the large `files/` folder. To rebuild the fixtures from the full files run `python scripts/make_test_data.py`; to run the integration tests against the full files set `OOTP_TEST_DATA=full`.
