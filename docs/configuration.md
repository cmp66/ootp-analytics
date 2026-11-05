# Configuration

Every setting is a toml file in `config/`, each with comments at the top. What each one controls and what to do after editing it:

| File | Controls | After editing |
| --- | --- | --- |
| `config/saves.toml` | Every save (league): `source` (`dump` = OOTP database dumps, `export` = in-game export), `ratings` (the ratings set, e.g. `Standard27`), `ballparks` (park csv), `home_share`, `scales` (a league with 1-10 ratings), `[<save>.levels."AAA"]` (age and pro-years limits, `max_players` and `pitchers` of the level, and the promotion ladder, top level first), `[<save>.team_names]` (team abbreviation to the name in the ballpark csv). | New league: workflow 2 of [workflows.md](workflows.md). Limits: reload the viewer. Ballparks or team names: `update_all.py`. |
| `config/update.toml` | The human leagues that `update_all.py` refreshes and the viewer shows: `models_from` (the training league of the same game version), `org` (your team abbreviation), `phase` (where the league is in the season: `season`, `offseason` or `spring`; the level moves page starts at the matching MLB active roster of 26 / 40 / 60), optional `steps`. The first save is the one the viewer opens. | Reload the viewer page. A new `org` also needs `update_all.py --only lists`. |
| `config/viewer.toml` | The lists (very good, good, ranking, draft, IAFA, free agents, trade targets, ...): metric, pool, whether depth players are hidden. | `update_all.py --only lists`, or look at one with `outlook_lists.py <save> --list <name>`. |
| `config/acquisition.toml` | The acquisitions page: what a need is (`need_percentile`, `target_percentile`), `bullpen`, `min_gain`, `hard_to_sign`, the weight of a reliever, and the priority list (`min_budget`, `multiyear_demand`, `multiyear_years`, `block_weight`, `non_need_factor`, `[signing]` chances, `[popularity.local]` and `[popularity.national]` boosts). | Reload the page; most of it can also be changed on the page for the session. A config-only change on the server needs `deploy/deploy.sh`. |
| `config/minors.toml` | The level moves page and the prospect tiers: the key ratings, what counts as green, the starters wanted per level (`starters`), the pitcher slack and the position gap, and `[mlb]`: the active roster of each phase (26 / 40 / 60), the pitchers (13) and starters (5) wanted in the season, and the option years, `prospect` / `top_prospect` ceiling percentiles, the last age at which ratings still grow (per game version). | Reload the page. |
| `config/rule5.toml` | The Rule 5 page: 40-man size and the export column (`ON40`), `ready_percentile`, `potential_percentile`. | Reload the page. |
| `config/outlook.toml` | Per ratings set: the minimum ratings to be ranked at a position, and the cut points that turn the dump's 1-200 trait numbers into the export's categories. Written by `build_outlook_tables.py` (the first time for a ratings set), edited by hand where needed. | Rebuild and retrain the outlook models if the minimums change (workflow 3). |
| `config/rating-scales/<version>.csv` | How often each 20-80 value is each 1-10 value, per rating column and game version. In git. | Only for a new game version: `build_scale_tables.py`. |

Rules of thumb:

- The loaders reject unknown keys, so a typo in a toml fails loudly instead of being ignored.
- Thresholds that are percentiles are a share of the league (0.40 = the lower 40 %), not a rating.
- Files are read when the viewer page loads; the data (csv) is cached, so after an update reload the page.
- The server has its own copy of `config/`, part of the image: see [updating-code.md](updating-code.md).
