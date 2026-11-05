# ootp-analytics: notes for Claude Code

Out of the Park Baseball analytics: value and outlook models, an update pipeline (`scripts/update_all.py`) and a Streamlit viewer
(`scripts/run_viewer.py`, also hosted on a Lightsail server). Start with `docs/workflows.md` (an "I want to ..." index), then
`docs/viewer.md` (every screen), `docs/configuration.md` (every `config/*.toml`), `docs/scripts.md` and `docs/utilities.md`.
The user reads the docs by workflow, not by script list: keep that structure when you document something.

## How we work

- **Branches.** One branch per piece of work, from an up to date `main`. The user opens and merges the PR on GitHub (there is no
  `gh` CLI). After a merge: `git checkout main && git pull`, delete the local branch.
- **Commits and pushes.** Leave finished work uncommitted until the user has reviewed it; commit when asked ("commit ...") and
  push only when asked. A commit message ends with the Co-Authored-By line given in the session. Never commit `files/`, `.env`
  or `.vscode/`; check `git status` before `git add -A`.
- **Before you say it is done:** `black` (line length 120), `flake8 --max-line-length 130 viewer model scripts tests util`,
  `python -m pytest` (about 3.5 minutes; do not run it while training: both use the CPU fully). Add a test with every change.
  `tests/test_viewer_memory.py` keeps every viewer page inside 500 MB: load data through `viewer/data_cache.py`, import nothing
  heavy (torch, scikit-learn) in the viewer.
- **Document as you go.** A change to a screen goes into `docs/viewer.md`, a new setting into `docs/configuration.md` and the
  toml comment, a new script into `docs/scripts.md` and the table in `docs/workflows.md`.
- **Say what is a guess.** Numbers tuned by hand (penalties, thresholds) are reported as such; the user adjusts them.
- Environment: Python 3.14, pyenv virtualenv `ootp-analytics`, run scripts as `PYTHONPATH=. python scripts/<name>.py`.

## Not in git, and how it travels between machines

`files/` (leagues, models, outputs), `plans/` (history of the larger pieces of work), Claude's memory notes and the conversations
are not in git. All of it goes through the S3 bucket `ootp-models` (access restricted to the user's IP addresses):

| What | Push | Pull |
| --- | --- | --- |
| trained models | `scripts/sync_models.py push` | `pull` |
| raw league files (exports) | `scripts/sync_files.py push` | `pull` |
| viewer files for the server | `deploy/push_data.sh` | (the server pulls) |
| memory notes, `plans/` (and `--set sessions` for the conversations) | `scripts/sync_notes.py push` | `pull` |

The user finishes a big item on one machine, pushes, and pulls on the other; everything is in `main` when they switch. A new
machine: `scripts/setup_machine.py` (workflow 6). Push `sync_notes.py` when you finish work that taught you something worth
keeping, and tell the user to pull on the other machine.

## The export (OOTP "person" lists in `files/<save>/<season>/ratings/`)

- Saves: TGS27 (human, CLE), Dugout26 (human, CIN), PBL27 (human, TOR); training leagues Model-Sim (OOTP 27) and
  Model-Sim-26-Std (OOTP 26) are database dumps, never scored. `files/TGS27` is a symlink into the Windows folder.
- Ratings are exported often; **finance and statistics only periodically**: do not flag them as stale.
- The TGS27 export must hold **all levels** (MLB and every minor league list); an MLB-only export empties the minor league pages.
- Columns the pages read: `R5` (draft eligible), `ON40` (on the 40-man), `ACT` (active roster), `WAIV`, `DFA`, `DEM` (a free
  agent's one-year demand; 0 = minor league deal; for draft and IAFA players the signing bonus), `Sign` (signability), `SLR`,
  `YL` (contract), `PROY`, `LPOS` (listed position), `Nat. Pop.` and `Loc. Pop.` (Unkn., Insig., Fair, Known, Pop., V.Pop,
  Ex.Pop.), and for TGS27 `OPT` (option years used, 0 to 3) and `OY` (times optioned this season). Other leagues: `OY` is the
  older field and there is no `OPT`, so no option information.
- After a new export: `update_all.py`, reload the viewer.

## Baseball rules the code follows

- **Rankings use ratings, not stats** (`value_now` and the outlook models); injury risk is its own column. Value models train on
  MLB-level players only; the low end (minor leaguers) is extrapolation.
- **Option years.** 3 per player. The first demotion of a season uses one (`OPT` +1, `OY` 1); once `OY` >= 1 he can go down and up
  freely that season. `OPT` 3 and `OY` 0: no option left, waivers (DFA) needed. `OPT` 2 and `OY` 0 still has one option year.
- **Active roster by phase:** 26 in the regular season and playoffs, 40 in the offseason, 60 in spring training. All 40-man players
  are on the active roster in spring training, so choosing 26 there simulates the cutdown. `phase` in `config/update.toml` says
  where each league is (CLE spring, CIN season, TOR offseason: change it as the game moves).
- **Level moves.** The majors are filled first with the best players (current value); AAA problems are solved from AA, never by
  sending a major leaguer down. Moves respect the age and pro-years limits of each level (a 26 year old does not go down to a level
  with maximum age 25), `max_players` and the `pitchers` target (a target, +-1, because pitchers get hurt) per level in
  `config/saves.toml`. INT (international complex) is `no_demotion`: players are placed there automatically. Winter leagues are
  `ladder = false` and not managed yet (a separate work item). Every level needs 2 catchers, 5 infielders, 4 outfielders and a
  rotation (6 SP in the minors, 5 in the majors; the MLB season roster is 13 pitchers: 5 starters, 8 relievers). AAA has no age
  limit and is the last stop for prospects and the cover for injuries; lower down prospect status counts for more and a filler is
  just a filler.
- **Acquisitions.** The priority list ranks by expected impact (runs over the contract against the starter and the own pipeline,
  times the chance he signs); a free agent asking $8m or more is assumed to want three years; keep $10m of budget space for the
  draft and IAFA; popularity boosts (local counts much more than national).
- **Rule 5.** Rebuilding teams draft on potential, so high-potential prospects are protected, and exposed high-potential prospects
  of other organizations are listed as draft candidates.

## Gotchas

- Streamlit page code runs on every rerun: heavy work goes in `st.cache_data` with `max_entries`, shared frames in
  `viewer/data_cache.py` (read only, 32-bit floats, two leagues at most).
- The viewer server is small (600 MB container): new code goes with `deploy/deploy.sh`, new data with `deploy/push_data.sh`.
- Pandas performance warnings are errors in the tests (fragmented frames fail).
- The code of the roster plan (`viewer/roster.py`) is an integer program (scipy `milp`); its penalties are hand tuned constants at
  the top of the file.
