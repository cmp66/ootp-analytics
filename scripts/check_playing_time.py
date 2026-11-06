"""Does actual playing time matter for a player's development? Hitters (plate appearances), pitchers (batters faced) and defense
(innings at infield, outfield and catcher).

Usage:
    PYTHONPATH=. python scripts/check_playing_time.py                         # the training leagues, every group
    PYTHONPATH=. python scripts/check_playing_time.py --saves Model-Sim TGS27 --groups hitters pitchers

For each pair of consecutive seasons the change of a player's ratings (the mean of the ratings of the group, in rating points,
between the two snapshots) is regressed on his age, level, previous rating, gap to potential, work ethic and intelligence, and on
his playing time, in units of 100 PA or batters faced, or 100 innings, with standard errors clustered by player. Two estimates:

  raw    the playing time of the season of the change alone. Managers give time to the players who improve (and the improvement of
         a player goes on from year to year), so this OVERSTATES the effect: an upper bound.
  with   the playing time of the season before and of the season after as well. The season AFTER cannot cause the change, so the
         size of its coefficient (the placebo) shows how much of the playing time is selection. But playing time is persistent and
         the season after carries the change, so holding it fixed pushes the coefficient of the season of the change down: a
         lower bound, which can come out negative.

The effect of playing time is between the two, nearer the lower one when the placebo is large. Cross-section data cannot settle it
(the manager reacts to the improvement as it happens); the verdict says whether the placebo shows selection and how large
the upper bound is. Needs three consecutive seasons of a save (two: only the raw estimate). Playing time comes from the dump
(all levels) of the training leagues, and from the processed hitting, pitching and fielding output of the other saves (the levels
exported).
Results of the training leagues and the reading: docs/models.md, "Playing time and development".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import data  # noqa: E402
from model.outlook_data import league_levels, playing_time  # noqa: E402
from util.seasons import available_seasons  # noqa: E402

HIT = ["CON", "GAP", "EYE", "K's", "POW", "BABIP"]
HIT_POT = ["CON P", "GAP P", "EYE P", "K P", "POW P", "BA P"]
PIT = ["STU", "MOV", "HRR", "PBABIP", "CON.1"]
PIT_POT = [f"{c} P" for c in PIT]
# group -> (playing time column, ratings of the change, potential ratings, who, scale of the playing time unit)
GROUPS = {
    "hitters": ("pa", HIT, HIT_POT, "hitter", 100),
    "pitchers": ("bf", PIT, PIT_POT, "pitcher", 100),
    "infield": ("fip_IF", ["IF RNG", "IF ARM", "IF ERR", "TDP"], [], "hitter", 100),
    "outfield": ("fip_OF", ["OF RNG", "OF ARM", "OF ERR"], [], "hitter", 100),
    "catcher": ("fip_C", ["C ABI", "C FRM", "C ARM"], [], "hitter", 100),
}
COLUMNS = (
    ["ID", "LPOS", "AgeFine", "WE", "INT", "league_id"]
    + HIT
    + HIT_POT
    + PIT
    + PIT_POT
    + [c for g in GROUPS.values() for c in g[1]]
)
FLAG_Z = 2.5


def clustered_ols(y, X, groups):
    """Least squares with standard errors clustered by `groups`: (coefficients, standard errors, rows, R2)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y, groups = X[ok], y[ok], np.asarray(groups)[ok]
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    clusters, index = np.unique(groups, return_inverse=True)
    score = np.zeros((len(clusters), X.shape[1]))
    np.add.at(score, index, X * resid[:, None])
    cov = bread @ (score.T @ score) @ bread * (len(clusters) / max(len(clusters) - 1, 1))
    return beta, np.sqrt(np.diag(cov)), len(y), 1 - resid.var() / y.var()


def innings_by_group(save: str, season: int) -> pd.DataFrame:
    """Innings of the season at catcher, infield and outfield per player, from the processed fielding output."""
    fielding = data.read_output(save, season, "fielding")
    group = fielding["POS"].map(
        lambda p: "C" if p == 2 else "IF" if p in (3, 4, 5, 6) else "OF" if p in (7, 8, 9) else "O"
    )
    return fielding.groupby(["ID", group])["IPClean"].sum().unstack(fill_value=0).add_prefix("fip_")


def season_playing_time(save: str, season: int) -> pd.DataFrame:
    """pa, bf (and primary_level where the save has it) per player ID: the dump of a training league, else the output."""
    found = playing_time(save, season)
    if len(found):
        return found[["pa", "bf"]]
    hitting = data.read_output(save, season, "hitting")
    pitching = data.read_output(save, season, "pitching")
    return pd.concat(
        [hitting.groupby("ID")["PA"].sum().rename("pa"), pitching.groupby("ID")["BF"].sum().rename("bf")], axis=1
    ).fillna(0)


def players_league(save: str, season: int, players: pd.DataFrame) -> pd.Series:
    """The level of the league each player is in at the snapshot (1 = MLB), NaN for a player without one."""
    levels = league_levels(save, season)
    return players["league_id"].map(levels) if "league_id" in players else pd.Series(np.nan, index=players.index)


def load_panel(save: str, seasons: list[int] | None = None) -> pd.DataFrame:
    """One row per player and season: ratings, traits, playing time (pa, bf, innings by position group) and level."""
    rows = []
    for season in seasons or available_seasons(save):
        players = data.read_output(save, season, "player-data").drop_duplicates("ID")
        players = players[[c for c in COLUMNS if c in players.columns]].set_index("ID")
        frame = players.join(season_playing_time(save, season)).join(innings_by_group(save, season))
        frame["level"] = pd.to_numeric(players_league(save, season, players), errors="coerce")
        frame["season"] = season
        rows.append(frame.reset_index())
    panel = pd.concat(rows, ignore_index=True)
    for column in ("pa", "bf", "fip_C", "fip_IF", "fip_OF"):
        panel[column] = panel.get(column, 0).fillna(0) if column in panel else 0.0
    if "level" not in panel:
        panel["level"] = np.nan
    return panel


def development_table(panel: pd.DataFrame, group: str) -> pd.DataFrame:
    """Rows of consecutive seasons of one group with the change of its ratings, the controls and the playing time of the season
    before (`prev_pt`), of the change (`pt`) and after (`next_pt`; NaN when the player has no next season)."""
    time_column, ratings, potentials, who, _ = GROUPS[group]
    ratings = [c for c in ratings if c in panel.columns]
    potentials = [c for c in potentials if c in panel.columns]
    wanted = panel["LPOS"] == "P" if who == "pitcher" else panel["LPOS"] != "P"
    table = panel[wanted].sort_values(["ID", "season"]).copy()
    table["cur"] = table[ratings].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    table["pot"] = table[potentials].apply(pd.to_numeric, errors="coerce").mean(axis=1) if potentials else 0.0
    by_player = table.groupby("ID")
    table["delta"] = table["cur"] - by_player["cur"].shift(1)
    table["prev_cur"] = by_player["cur"].shift(1)
    table["prev_gap"] = by_player["pot"].shift(1) - by_player["cur"].shift(1)
    table["prev_age"] = by_player["AgeFine"].shift(1)
    table["prev_level"] = by_player["level"].shift(1)
    table["pt"] = table[time_column]
    table["prev_pt"] = by_player[time_column].shift(1)
    table["next_pt"] = by_player[time_column].shift(-1)
    table["gap_years"] = table["season"] - by_player["season"].shift(1)
    table = table[table["gap_years"] == 1]
    if group in ("infield", "outfield", "catcher"):  # players who played the position at some time around the change
        table = table[(table["pt"] > 0) | (table["prev_pt"] > 0) | (table["next_pt"] > 0)]
    return table


def design(table: pd.DataFrame, terms: list[str], unit: float) -> pd.DataFrame:
    age = pd.get_dummies(table["prev_age"].clip(17, 36).round().astype("Int64"), prefix="age", dtype=float)
    level = pd.get_dummies(table["prev_level"].astype("Int64"), prefix="level", dtype=float)
    traits = pd.DataFrame(
        {
            "WE": pd.to_numeric(table["WE"], errors="coerce") / 100,
            "INT": pd.to_numeric(table["INT"], errors="coerce") / 100,
        }
    )
    controls = table[["prev_cur", "prev_gap"]].astype(float)
    return pd.concat([age, level, controls, traits] + [(table[t] / unit).rename(t) for t in terms], axis=1)


def estimate(table: pd.DataFrame, unit: float = 100, terms=("pt", "prev_pt", "next_pt")) -> dict:
    """The coefficients (rating points per `unit` of playing time) and standard errors of `terms`, plus the rows used."""
    terms = list(terms)
    if "next_pt" in terms:
        table = table[table["next_pt"].notna()]
    table = table[table["prev_pt"].notna()] if "prev_pt" in terms else table
    X = design(table, terms, unit)
    beta, se, rows, r2 = clustered_ols(table["delta"], X, table["ID"])
    names = list(X.columns)
    return {"rows": rows, "r2": r2, **{t: (beta[names.index(t)], se[names.index(t)]) for t in terms}}


def verdict(
    raw: tuple[float, float], placebo: tuple[float, float] | None, unit: float = 100, season: float = 600
) -> str:
    """What the estimates say: the upper bound as a change per `season` of playing time, and whether the placebo
    (the season after) shows selection (a positive coefficient of at least FLAG_Z standard errors)."""
    per_season = f"upper bound {raw[0] * season / unit:+.2f} points per {season:.0f}"
    if placebo is None:
        return f"{per_season}, no placebo"
    selected = placebo[0] > 0 and placebo[0] / placebo[1] >= FLAG_Z
    return f"{per_season}; placebo {'shows selection' if selected else 'is flat'}"


def report(saves: list[str], groups: list[str]) -> None:
    for save in saves:
        seasons = available_seasons(save)
        print(
            f"\n== {save}: {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})"
            if seasons
            else f"\n== {save}: no seasons"
        )
        if len(seasons) < 2:
            continue
        panel = load_panel(save)
        for group in groups:
            unit = GROUPS[group][4]
            table = development_table(panel, group)
            if table.empty:
                print(f"  {group}: no rows")
                continue
            raw = estimate(table, unit, ("pt",))
            full = estimate(table, unit) if len(seasons) >= 3 else None
            b, se = raw["pt"]
            line = f"  {group:9} rows {raw['rows']:7}  per {unit} {GROUPS[group][0]}: raw {b:+.3f} ({se:.3f})"
            if full:
                (b2, se2), (b3, se3) = full["pt"], full["next_pt"]
                line += f"  with {b2:+.3f} ({se2:.3f})  placebo {b3:+.3f} ({se3:.3f})"
            print(line + "  -> " + verdict(raw["pt"], full["next_pt"] if full else None, unit))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--saves", nargs="+", default=["Model-Sim", "Model-Sim-26-Std"])
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = parser.parse_args(argv)
    report(args.saves, args.groups)


if __name__ == "__main__":
    main()
