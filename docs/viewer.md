# The viewer

`python scripts/run_viewer.py` starts it at http://localhost:8501 (`--port` changes it). It only reads the csv files `update_all.py` writes (see [updating-data.md](updating-data.md)); reload the page after an update. On a server it runs in a container: [deploy/README.md](../deploy/README.md).

## Using it

- **League:** the selector at the left of the top row of every page switches between the saves of `config/update.toml`; the choice follows to the other pages. Each league opens on its own `org`, and the first save of the file is the one the viewer opens.
- **Navigation:** a row of links at the top of every page.
- **Values:** the number on most pages is a player's current value (`value_now`): runs above the league's average starter at the position (offense + baserunning + defense add up to it). Ceiling and floor are the 90th and 10th percentile of his best season within the window.
- **Player page:** click a cell of a player's row in any table (the explorer lists, the acquisition tables, the level-moves tables); the depth chart, which is html, has a list of its players under the filters that does the same.
- **Park-adjusted values toggle** (organization, minor leagues, draft, IAFA, depth chart, explorer, acquisitions): values every player as if half his games were played in the park of the organization you are viewing (`home_share` of the save). Home runs, doubles, triples and hits move with the park factors (the left or right one by the side he bats from; singles are what is left of the hits) at the league's run weights, a pitcher's runs allowed with the park's run factor. Fielding and baserunning stay, the chances of being a top-10 or top-50 player are not moved, and a `Park` column shows what the park changes. On the acquisitions page every team's starters are valued in their own park and the candidates in yours. The toggle is off for all organizations and free agents and for leagues without park factors or park models. The hitters' park rates come from `predict_park.py` (the `park` step of `update_all.py`, `viewer/parks.py`).

## The screens

### Organization overview
One dense screen for an organization (the selector also has all organizations and the free agents): the best 20 position players, the best 10 at each position, the best 20 starting and 20 relief pitchers, by current value. Its only choice is the organization.

### Minor leagues
The same layout for players not on the MLB roster, with a Ceiling / Floor switch instead of the current value: the best 90th (or 10th) percentile value within 5 seasons (or the career) and the best chance of becoming a top-10 (or top-50) player at the position.

### MLB draft and IAFA
The same layout for the draft class and the international amateurs. They belong to no organization yet, so there is no selector.

### Depth chart
An OOTP-style field for one organization: the best five at each fielding position and at designated hitter, and the best ten starting and relief pitchers. It is built the way a manager would: the best lineup first, with every player at the position where he adds the most given the others (an assignment problem that maximizes the total value, so the second-best center fielder who is better at right field than anyone else plays right field), then the best backups from the players left; pitchers likewise; each player once.

- **Show:** the entire organization, the MLB roster alone, the minor leagues, or one level / team (AAA, AA, A+, A-, R+, R, the International Complex, ...).
- **Ranked by:** current value, or the ceiling / floor of his best season within 10 years.
- **Lineup** (Overall, vs RHP, vs LHP; current values only): the hitters against right- or left-handed starting pitchers. A hitter's value against a side is his value now with the batting part swapped for his split (`batting_vs_right` / `batting_vs_left` of the predicted values); fielding and running are the same, and the park effect uses the side a switch hitter bats from against that side. The pitchers stay one list.
- **Red names** are players over the age or pro-years limit of their level.
- Limits: `config/saves.toml`, one table per level (`[TGS27.levels."AA"]` with `max_age` and `max_pro_years`; 0 = no limit; the years come from the ratings column `PROY`). Reload the page after editing. Code: `viewer/depth.py`.

### Level moves (minor league management)
Per level of your organization, the players who should move up or down. It reads each player's key ratings (hitters: contact, gap, power, eye, avoid K's, BABIP; pitchers: stuff, movement, control) against all players of his level in the league, the way you read the game's ratings relative to a level; statistics are not used.

- **Up:** green in enough key ratings and not at the bottom of the next level, or old for the level with average ratings (no stagnating).
- **Must move:** over the age or pro-years limit of his level.
- **Down:** weak for the level, and only to a level whose limits he meets.
- **Focus:** the better players while they can still develop. A prospect has a ceiling in the better fifth of the league's minor leaguers of his kind and a development window (what is left until the age where ratings stop growing: hitters 25, pitchers 26, set per game version). Depth players and players past their window are listed only on request.
- Rules: `config/minors.toml` (changeable on the page for the session); limits and the promotion ladder (the order of the level tables, top first; `ladder = false` leaves a level out): `config/saves.toml`. Check the ladder order of a new league.

### Acquisitions (MLB roster)
Where the starting lineup, rotation and bullpen of your organization are weak against the starters of the league, and who could upgrade it.

- **Needs:** the best lineup of the MLB players is chosen (one per position), five starters and `bullpen` relievers; each starter is placed among the league's starters at his place. A place nobody can play (a missing catcher) is the biggest need.
- **Candidates:** your own players who are not starters (internal: the bench and the minor leaguers), players on waivers (claimed with their contract) and free agents (their one-year demand, signability shown). A demand of 0 is a minor league deal.
- **Cost** is against the budget space of the season (from the `finance` folder of the export, `<save>-<season>-finances.csv`); without it the costs are not checked.
- **Priority list** (the top table): the acquisitions in the order to act.
  - *Impact* is the runs a player adds over his contract (three years at most) against the starter and against what your own prospects at the place will be worth in one and two years. A year in which he would keep a ready prospect out of the lineup counts against him. *Per yr* is Impact divided by the years.
  - *Expected* is Impact times the chance he signs (by signability: the demand can change before he does) times a popularity boost. Local popularity counts most (the local market, fan interest, which matters when yours is low): +10% for a very popular (V.Pop) and +20% for an extremely popular (Ex.Pop) player. National popularity (awards, all-star votes) counts little: +2% and +4%; the two add up. Columns *Local pop* and *Nat. pop*. It is a tiebreaker, since it scales an impact that is small for a player who adds little. The list is sorted by Expected. *Fallbacks* are the next options at the place.
  - *Verdict:* **Hole** = nothing of your own reaches the league target within two years, so a long deal is fine; **Bridge** / **Soon** = a prospect reaches it in one / two years, so prefer a short deal; **Fill** = the place is no need. A free agent asking `multiyear_demand` or more ($8m) is assumed to want three years.
  - *Spend:* the list never takes the budget space under `min_budget` ($10m, kept for the draft, IAFA and other areas). **Within** fits, **Over** would take it under the minimum after the deals above it.
  - A good player can cover two places as a backup, so one of your players may be counted as the future answer at more than one.
- **Best set of acquisitions:** the largest total gain that fits the budget space (an integer program: one acquisition per place, each player once), and a table of candidates for any place.
- Rules: `config/acquisition.toml` (changeable on the page). The export columns it reads: `WAIV` (on waivers), `DEM` (demand), `Sign` (signability), `SLR` and `YL` (contract). Code: `viewer/acquire.py`.

### Rule 5
Prepares the draft. Your draft-eligible players (the `R5` flag of the export) with a recommendation:

- **Protect** (on the 40-man) when he is a prospect with development left whose ceiling would start for your MLB team (or a top prospect) and he is ready for the majors, when he is ready and would start now, or when his potential is high (ceiling in the better tenth of the league's minor leaguers at his position: a rebuilding team drafts on potential and does not mind the roster spot).
- **Low risk** when he is a prospect who is not ready, so unlikely to be picked. **Expose** for the rest.
- **40-man arithmetic:** on it today (the `ON40` flag; a league without it gets an estimate, marked on the page), open slots, the players you protect, the free agents you plan to sign, how many have to come off.
- **Who could come off:** the players with the least to keep them (not starters, no prospects, lowest value), with their contracts.
- **Draft candidates:** draft-eligible players of other organizations who would improve your MLB roster, and a last table with the exposed prospects of other organizations with that much potential whether or not they help now (for a weak season).
- Rules: `config/rule5.toml`. Code: `viewer/rule5.py`.

### Explorer
A list builder with many options (list, seasons ahead, sort, players from, positions, age, levels, organization) and an Excel report with the ordered list, a block per position and the filters used. Code: `viewer/queries.py`, `viewer/overview.py`, `viewer/report.py`; the lists are defined in `config/viewer.toml` (also `outlook_lists.py`).

## Adding or changing a screen

See [updating-code.md](updating-code.md): load the data through `viewer/data_cache.py`, register the page in `viewer/app.py`, and the memory test and the page test have to pass.
