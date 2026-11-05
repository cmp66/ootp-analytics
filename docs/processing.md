# How the data is processed

Reference for what `process_seasons.py` computes: park adjustment, fielding, the ability values the models learn from. For the order in which to run things see [workflows.md](workflows.md).

## Processing a save

`python scripts/process_seasons.py <save>` reads the raw files of a save (a folder under `files/` with a table in `config/saves.toml`, which sets the source, ratings set and ballpark file) and writes the processed csv files to `files/<save>/<season>/output/`. By default it processes the latest season only; `--season N` does one season, `--start N` runs from N through the latest, `--start N --stop M` runs the range, and `--list` shows the seasons that have data. A failing season is reported and the rest still run (exit status 1). This replaces `dataImport.ipynb`.

## Park factors

`OFFAdj` (batters) and `WAAAdj` (pitchers) are park adjusted when a save lists a ballpark ratings csv in `config/saves.toml` (`ballparks = "ballpark_ratings.csv"`, kept in `files/<save>/`; `files/<save>/<season>/ballpark_ratings.csv` overrides it for one season). Without it they equal `OFF` and `WAA`. Columns: `Team Name, Park, PF AVG, AVG L, AVG R, PF HR, HR L, HR R, PF D, PF T, PF`, where 1.000 is neutral and the L / R columns are for left- and right-handed batters.

- `home_share` (default 0.5) is the share of a player's plate appearances / batters faced in his own park; the rest count as neutral.
- Team names in the csv are matched to the save's teams: dump saves use "city nickname" automatically; export saves need a `[<save>.team_names]` table (team abbreviation to csv name); any exception goes in the same table. Teams without a park are logged and treated as neutral.
- Batters: hits, doubles, triples and home runs are divided by the park factor (by batting side), then `wOBAAdj` and `wRAAAdj` are computed against the league mean of the adjusted wOBA, so league `wRAAAdj` sums to 0. New columns `ParkAVG`, `ParkHR`, `ParkD`, `ParkT`, `wOBAAdj`, `wRAAAdj`. `OFFAdj = BSR + wRAAAdj`.
- Pitchers: expected runs use the run factor `PF`; new columns `ParkRuns`, `ExpRA9Adj`; `WAAAdj` still sums to 0 in every usage bucket.

## Fielding

The make-rate fielding columns (`wPM*`, `wRunsPSeason`, `runsPAdj`, `runsPAdjSeason`, `PlayPercentAA`) judge a fielder against the league **inside each BIZ difficulty tier** (R, L, E, U, Z; "I", impossible, is ignored). `PlaysExpected` is what an average fielder at his position makes on his chances, `PlaysAboveExpected` is the difference and `PlayPercentExp` the expected make rate; the league tier rates are written to `<save>-<season>-fielding-tier-rates.csv`. This removes the effect of the staff and park on which kinds of balls a fielder sees (pitcher and batter BABIP ratings were checked and have no measurable effect). Errors in these columns are charged per chance.

The volume columns (`PlayMadeAA*`, `PlayAAASeason`, `runsPSeason`, `runsPNow`, `EAASeason`) count plays per inning, so they also depend on how many balls in play the staff allows (mostly its strikeout rate). `scripts/validate_fielding_ability.py` and `scripts/fielding_attribute_report.py` check how well the values isolate the fielder.

## Ability values for the models

Fielding: `FldRunsSeason` / `FldRunsAdj` (runs: tier-adjusted plays, catcher framing, outfield arm) are the fielding value without the noisy errors; `ErrRunsSeason` / `ErrRunsAdj` are the error charge on its own. Pitching: `WAA` splits into `FIPRuns` (home runs, walks and hit batters, strikeouts against the usage-bucket average, 13 / 3 / -2 weights), `BABIPRuns` (hits on balls in play against the league BABIP; mostly luck and defense, use `BIP` as the weight) and `OtherRuns` (sequencing, unearned runs, defense): `WAA x 10 = FIPRuns + BABIPRuns + OtherRuns`. `scripts/validate_ability_targets.py` reports how repeatable each value is.

