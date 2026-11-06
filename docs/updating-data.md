# Updating the data

How new data gets from the game to the viewer, on your machine and on the server. For the order of the other jobs (retraining, a new league) see [workflows.md](workflows.md).

## The short version

```
1. export from OOTP into files/<save>/<season>/           (ratings often; finance and statistics now and then)
2. PYTHONPATH=. python scripts/update_all.py              (about 35 s for two saves)
3. reload the viewer (python scripts/run_viewer.py)
4. deploy/push_data.sh                                    (only if the server should show it too)
```

## What to export

A human league save (`source = "export"` in `config/saves.toml`) lives in `files/<save>/<season>/`:

| What | Where | How often | Used by |
| --- | --- | --- | --- |
| Player lists: MLB, all free agents, international amateur free agents, draft pool (the *person* exports) | `ratings/` | Often: after every sim period you want to look at | Everything: values, outlook, lists, every screen |
| Finance (the league's team finances) | `finance/` | Now and then | The acquisitions page (budget space, payroll, strategy) |
| Statistics (team batting, pitching, fielding; minor league stats) | the season folder | Now and then | The processed statistics and run values |
| Ballpark ratings | `files/<save>/ballpark_ratings.csv` (or `<season>/ballpark_ratings.csv` for one season) | When parks change | Park adjustment (only if `ballparks` is set in `config/saves.toml`) |

- **Finance and statistics are updated only periodically.** A page that shows older finance or statistics than the ratings is normal, not a mistake in the update.
- **The player lists must hold every level.** The MLB list and every minor league list have to be in `ratings/`. An export with the MLB only makes the minor league pages empty. If a page looks empty at AAA / R, first check that the minor league lists are there.
- **Columns the pages read** (add them to the export view if they are missing): `R5` (draft eligible) and `ON40` (on the 40-man roster) for the Rule 5 page, `OY` (options used), `WAIV` (on waivers), `DFA`, `DEM` (a free agent's one-year demand; for IAFA and draft players the signing bonus), `Sign` (signability), `SLR` and `YL` (contract), `PROY` (pro years, for the level limits), `OPT` (option years used, 0 to 3) and `OY` (times optioned this season) for the level moves page, `INJ` and `Left` (injured now, and the time left, so the level moves page does not move injured players), `Nat. Pop.` and `Loc. Pop.` (popularity, for the acquisitions priority list; values Unkn., Insig., Fair, Known, Pop., V.Pop, Ex.Pop.). A demand of 0 means a minor league deal. After adding a column, re-export the player lists and run `update_all.py`.

## Everyday: new ratings files

1. `copy_exports.py` copies the export into the save's latest season folder (same layout as the earlier seasons); see [scripts.md](scripts.md). Or put the files there yourself.
2. `update_all.py` (all saves of `config/update.toml`) or `update_all.py TGS27`. It runs, per save: process the seasons, score the values, score the outlook, write the park rates and the lists of `config/viewer.toml` into `output/lists/`.
   - `--season 2046` when it is not the latest season in the folder.
   - `--if-changed` skips a save whose input files have not changed since the last update.
   - `--only outlook lists` redoes some steps; `--steps` lists their names.
3. A step that fails is named in the summary at the end and the exit status is 1; the other saves still run.
4. Reload the viewer page.

`update_all.py` never trains. If the lists that compare with your roster are missing, set `org` for the save in `config/update.toml`.

## A new season of the same league

Create `files/<save>/<next season>/` with the same layout (ratings, and finance / statistics when you have them) and run `update_all.py`. The viewer shows the latest season that has outlook files. Statistics of the new season will be thin early on; the ratings drive the values.

## Another league or another machine

- **A new league:** add it to `config/saves.toml` and `config/update.toml` (workflow 2 of [workflows.md](workflows.md)); if its ratings are not all 20-80, workflow 2b.
- **Another machine:** raw files travel through S3. On the machine that has the export: `python scripts/sync_files.py push`; on the other: `sync_files.py pull`, then `update_all.py`. The processed files are made again, never copied. The S3 bucket has to allow the machine's IP address. Details: [scripts.md](scripts.md), `sync_files.py`.

## Getting it to the viewer on the server

The server shows only the processed files of the latest season of each human league.

```
PYTHONPATH=. python scripts/update_all.py
VIEWER_HOST=ubuntu@<ip> deploy/push_data.sh
```

`push_data.sh` pushes the `viewer` set to S3 (`sync_files.py push --set viewer`, about 160 MB for three leagues, only the changed files go) and runs `deploy/refresh.sh` on the server over ssh. `refresh.sh` pulls what changed and restarts the viewer only when something changed (the viewer keeps the data in memory, so a restart is what shows new files). A cron job on the server can run `refresh.sh` on its own. Set `VIEWER_HOST` (and `SSH_KEY`) in your shell profile. One-time setup of the server: [deploy/README.md](../deploy/README.md).

## Updating the models

Not part of a data update. New training seasons, a processing fix or a new game version: workflow 3 or 4 of [workflows.md](workflows.md). After the models change, `update_all.py` so the human leagues use them, and `sync_models.py push` to keep the set.

## When something looks wrong

| Symptom | Check |
| --- | --- |
| A page shows the old numbers | Did `update_all.py` finish without a failed step? Reload the page. On the server: did `push_data.sh` / `refresh.sh` run? |
| Minor league, level moves or depth chart pages are empty or thin | The minor league player lists are missing from `ratings/` (see above). |
| Rule 5 shows an estimated 40-man | The export has no `ON40` column. |
| No budget space on the acquisitions page | The `finance` folder is missing from the season folder. |
| Park toggle disabled | No ballpark csv for the save, or no park models for its model set (`train_models.py --streams park`). |
| `files/<save>` is a symlink | Fine for scripts and the viewer. A container needs the real folder (mount the folder the link points to). |
| A step says the models are missing | `sync_models.py pull` (workflow 6), or train them (workflow 3). |
