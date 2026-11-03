# Outlook columns: what `scripts/predict_outlook.py` writes

`PYTHONPATH=. python scripts/predict_outlook.py <save> [--season N] [--models-from <save>]` scores every player of a season and writes
two files next to the other outputs of that season. It works for a dump save and for an in-game export, with the models of the
same game version (`--models-from Model-Sim-26-Std` for TGS26, `--models-from Model-Sim` for TGS). Models and the position
minimums come from the long CPU leagues (plan 16); all values are **relative to the league that is scored**, so a player is
compared with that league's average, and the tier chances are rescaled to that league (below).

## `<save>-<season>-outlook-slots.csv`: one row per player and ranked slot
Slots are C, 1B, 2B, 3B, SS, LF, CF, RF, DH (hitters) and SP, RP (pitchers). A player is ranked at a position when his key ratings (or
potential) pass the minimum in `config/outlook.toml`; DH and RP have no minimum.

| Column | Meaning |
|---|---|
| `ID`, `season`, `slot` | the player and the slot |
| `value_now` | rating-based value now (batting + baserunning + fielding at the position per season for hitters; DH batting + baserunning; pitching runs per 200 IP for pitchers), league relative |
| `pct_now`, `rank_now`, `tier_now` | percentile, rank (1 = best) and tier (top10, top50, rest) among the players ranked at the slot; ranks 10 and 50 are for 30 teams and scale with the number of teams |
| `value_h1` ... `value_h10` | median value 1, 2, 3, 5, 7 and 10 seasons ahead |
| `value_h99`, `floor_h99`, `ceiling_h99`, `p_top10_h99`, `p_top50_h99` | the career view: the median, 10th and 90th percentile of his best season (best value at the slot) within the next 10 seasons, and the chance that he is top 10 / top 50 at the slot at some point in those 10 seasons; `rank_score_h99` blends them. It has no stay, availability or season-value columns |
| `floor_h`, `ceiling_h` | 10th and 90th percentile of the value (an 80 % band, widened so that it covers 80 % of held-out outcomes); `q25_h`, `q75_h` the quartiles |
| `p_top10_h`, `p_top50_h` | chance of being top 10 / top 50 at the slot in that season (a player who will be gone counts as no): the average of a bagged classifier and the chance that his value distribution is above the league's own rank-10 / rank-50 value at the slot (see below) |
| `rank_score_h` | 0.25 floor + 0.5 median + 0.25 ceiling: the order for the "very good players" and "ordered ranking" lists |
| `season_value_h`, `season_floor_h`, `season_ceiling_h` | the same value once availability is taken into account (below) |
| `injury_cost_h` | runs of value per season lost to proneness against a normal-proneness player of the same profile (positive values only) |

## `<save>-<season>-outlook-players.csv`: one row per player
| Column | Meaning |
|---|---|
| `Age`, `LPOS`, `level` | age, listed position, numeric level (1 MLB, 2 AAA, 3 AA, 4 A, 5 low A (OOTP 26), 6 rookie, 0 free agent) |
| `best_value`, `best_pct` | best value over his ranked slots and its percentile in the league |
| `depth_player` | best value below the 25th percentile of the league (`--depth-percentile`): roster filler rather than a prospect |
| `p_stay_h` | chance he is still in the league that many seasons later |
| `avail_p10_h`, `avail_h`, `avail_p90_h` | share of a full season he plays (10th / 50th / 90th percentile); includes benching and call-ups, so read it as relative |
| `avail_normal_h` | the median availability of the same player with normal proneness |
| `avail_ratio_h`, `injury_risk_h` | own / normal availability (capped at 1.3) and `1 - ratio`: positive = injury prone, negative = more durable than normal |

## Comparing an injury-prone player with a durable one
Both have a `value_h`, `floor_h` and `ceiling_h` (the rate when playing). Compare `season_value_h`, `season_floor_h` and `season_ceiling_h`
instead: they multiply a positive value by the ratio of the player's availability to a normal player's, drawing the two uncertainties
independently, so a fragile player of the same rate has a lower median and a lower, wider band. A value below the league average
is left as it is (missing time costs nothing against the average). `injury_cost_h` is the median difference.

## Lists and comparisons with an organization (`scripts/outlook_lists.py`, `viewer/queries.py`, `config/viewer.toml`)
`PYTHONPATH=. python scripts/outlook_lists.py <save> --list <name> [--org ABBR] [--horizon 3] [--slot SS] [--top 25] [--out file.csv]`
- `very_good`: best chance to be a top-10 player at the position; `good`: best chance to be 11th to 50th (top 50 minus top 10); `ranking`: ordered by
  `rank_score` (one row per player, his best slot); the depth players are hidden in the first two.
- Pools: `draft` (export `SctCat` = Amateur: the draft class, present only in exports taken before the draft), `iafa` (`SctCat` = International, free
  agents up to age 19), `free_agents` (the other `Lev` = FA), `trade_targets` (players of other organizations), `my_depth` (my organization).
  A dump has no draft class and no IAFA pool.
- `p_start_h`: chance that he is still in the league and his value in h seasons is above the value now of the player my organization would have to
  replace to start at that slot (my best at C..RF and DH, my 5th best starter, my 7th best reliever); `p_roster_h`: the same against the backup
  place (one more). My roster is the organization's MLB-level players, which is the 26-man in an export (828 MLB-level players for 32 teams). The
  chance is taken from the quantiles `floor / q25 / median / q75 / ceiling` of his value. It is a depth-chart approximation from ratings, not the lineup.

## Two points about leagues
- **Tier chances are made league-relative in two ways.** The classifiers learned the value scale of the CPU leagues; in a human league the top
  end is lower (the 10th-best starter in TGS26 is worth 11.9 runs against 20.1 in the matching CPU league), which made raw chances too small
  there. (1) For each slot, tier and horizon the classifier's chances are rescaled to add up to the number of future top players already in
  the league (measured in the training league, scaled by the number of teams); (2) the other half of the chance is read from the player's value
  distribution against the league's own rank-10 / rank-50 value at the slot. The average of the two beat either alone on held-out seasons in both
  versions. Values, floors and ceilings are not rescaled.
- **Ranks and percentiles are measured against the organizations' players only.** The exports also list free agents and amateurs, which the
  models were not trained on; they are placed on the organizations' scale so that a percentile means the same in every league.
- **Only the current season** is scored from its player-data; no earlier season or history is needed, which is why the same models can be
  used on an export with a single season.
