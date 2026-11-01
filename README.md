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
