"""Who plays at each level: the lineup, the rotation and the bench of every level of a plan, by the playing value of
`viewer/playing.py`. Pure functions over the table of `roster.plan` (every player with the level he is at after the plan), the
outlook frame and the ratings, no UI.

Stage 2 of plans/17: the levels are those of the plan; here the players of a level are placed in its slots (one player per
lineup slot C, 1B, 2B, 3B, SS, LF, CF, RF, DH; the starters wanted in the rotation; the rest of the hitters on the bench and of
the pitchers in the bullpen) so that the sum of their playing values is the largest, and compared with the placement of the depth
chart (the most current value per position, `depth.fill`) that the level moves page names wrong positions by.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from util.config import LevelLimits
from viewer import depth, minors, playing, roster

LINEUP = roster.LINEUP
IMPOSSIBLE = 1e6


def standings(frame: pd.DataFrame) -> pd.DataFrame:
    """Where each player stands at each slot (0 to 1: the share of the league's rostered players at the slot below him by value
    now), one column `std_<slot>` per slot, indexed by ID."""
    rows = frame[frame["slot"].isin(depth.CHART_SLOTS)]
    if "org" in rows:
        rows = rows[~rows["org"].isin(["FA", "nan", ""])]
    rows = rows.drop_duplicates(["ID", "slot"]).assign(std=lambda r: r.groupby("slot")["value_now"].rank(pct=True))
    return rows.pivot(index="ID", columns="slot", values="std").add_prefix("std_")


def primary_of(table: pd.DataFrame) -> pd.Series:
    """The position a player is being trained at: the one he is listed at when it is a field position, else his best one."""
    listed = table["Listed"] if "Listed" in table else pd.Series("", index=table.index)
    best = table["Pos"] if "Pos" in table else pd.Series("", index=table.index)
    return listed.where(listed.isin(playing.FIELD), best)


def inputs(
    table: pd.DataFrame, frame: pd.DataFrame, ratings: pd.DataFrame, growth: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The input table of `playing.slot_utilities` for the players of the plan: the plan's columns (ID, Age, pitcher, Window,
    ceiling_pct), the primary position as `slot`, the standing at every slot, and the rating and potential at every field
    position. `growth` (ID, d_cur, d_pot) is the change of the ratings over the last year, when there is a history.
    """
    out = table[["ID", "Age", "pitcher", "Window", "ceiling_pct"]].copy()
    out["slot"] = primary_of(table).to_numpy()
    out = out.merge(standings(frame), left_on="ID", right_index=True, how="left")
    wanted = {}
    for position in playing.FIELD:
        wanted[f"rating_{position}"] = position
        wanted[f"pot_{position}"] = f"{position} Pot"
    have = ratings.drop_duplicates("ID").set_index("ID")
    for name, column in wanted.items():
        out[name] = pd.to_numeric(out["ID"].map(have[column]), errors="coerce") if column in have else np.nan
    if growth is not None and not growth.empty:
        out = out.merge(growth[["ID", "d_cur", "d_pot"]], on="ID", how="left")
    return out.reset_index(drop=True)


def median_ages(frame: pd.DataFrame, table: pd.DataFrame) -> dict[str, float]:
    """The median age of the players of every level in the league (the organization's own when the league has none)."""
    rows = frame.drop_duplicates("ID")
    if "org" in rows:
        rows = rows[~rows["org"].isin(["FA", "nan", ""])]
    medians = rows.groupby("team_level")["Age"].median().to_dict() if {"team_level", "Age"} <= set(rows.columns) else {}
    for level, ages in table.groupby("Level")["Age"].median().items():
        medians.setdefault(level, float(ages))
    return medians


def _hitters(group: pd.DataFrame, utility: pd.DataFrame) -> list[tuple]:
    """(ID, slot) for the hitters of a level: the nine lineup slots by the largest total value, the rest on the bench."""
    hitters = group.loc[~group["pitcher"].astype(bool), "ID"].tolist()
    if not hitters:
        return []
    total = utility[utility["slot"].isin(LINEUP)].pivot(index="ID", columns="slot", values="total").reindex(hitters)
    total = total.reindex(columns=list(LINEUP))
    cost = np.where(total.isna(), IMPOSSIBLE, -total.fillna(0.0).to_numpy())
    rows, cols = linear_sum_assignment(cost)
    placed = {hitters[r]: LINEUP[c] for r, c in zip(rows, cols) if cost[r, c] < IMPOSSIBLE}
    return [(i, placed.get(i, "BENCH")) for i in hitters]


def _pitchers(group: pd.DataFrame, utility: pd.DataFrame, wanted: int) -> list[tuple]:
    """(ID, slot) for the pitchers of a level: the `wanted` who add most as a starter rather than in the bullpen; rest relieve."""
    pitchers = group.loc[group["pitcher"].astype(bool), "ID"].tolist()
    if not pitchers:
        return []
    total = utility[utility["slot"].isin(playing.PITCHER_SLOTS)].pivot(index="ID", columns="slot", values="total")
    total = total.reindex(index=pitchers, columns=list(playing.PITCHER_SLOTS))
    gain = (total["SP"] - total["RP"].fillna(0.0)).dropna().sort_values(ascending=False)
    starters = set(gain.index[:wanted])
    return [(i, "SP" if i in starters else "RP") for i in pitchers]


def lineups(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    ratings: pd.DataFrame,
    levels: dict[str, LevelLimits],
    rules: minors.Rules,
    settings: playing.Playing | None = None,
    growth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The slot of every player of the plan at the level he is at after it (`Final`; a player who leaves the organization is not
    placed). Columns: Level, Slot, ID, Player, Age, Primary (the position he is trained at), development, learning, hit, current,
    total, Reason and Old (the slot the depth chart's placement by current value gives him, "" when it gives him none).
    """
    settings = settings or playing.load_playing()
    rungs = minors.ladder(levels)
    if minors.TOP in set(table["Final"]):
        rungs = [minors.TOP, *rungs]
    kept = table[table["Final"].notna() & table["Final"].isin(rungs)]
    players = inputs(kept, frame, ratings, growth)
    ages = median_ages(frame, kept)
    names = kept.set_index("ID")["Player"]
    rows = []
    for level, group in kept.groupby("Final"):
        place = rungs.index(level)
        mine = players[players["ID"].isin(group["ID"])]
        utility = playing.slot_utilities(mine, ages.get(level, float(group["Age"].median())), place, settings)
        pool = frame[frame["ID"].isin(group["ID"]) & frame["slot"].isin(LINEUP)]
        old = {i: slot for slot, ids in depth.fill(pool, "value_now", dict.fromkeys(LINEUP, 1), 1).items() for i in ids}
        placed = _hitters(group, utility) + _pitchers(group, utility, roster._starters(level, rules))
        by_key = utility.set_index(["ID", "slot"])
        primary = dict(zip(mine["ID"], mine["slot"]))
        for player_id, slot in placed:
            part = by_key.loc[(player_id, slot)] if (player_id, slot) in by_key.index else None
            rows.append(
                (level, slot, player_id, names[player_id], float(group.loc[group["ID"] == player_id, "Age"].iloc[0]),
                 primary.get(player_id, ""), *(float(part[c]) if part is not None else np.nan for c in
                 ("development", "learning", "hit", "current", "total")), part["reason"] if part is not None else slot,
                 old.get(player_id, ""))
            )  # fmt: skip
    columns = [
        "Level",
        "Slot",
        "ID",
        "Player",
        "Age",
        "Primary",
        "development",
        "learning",
        "hit",
        "current",
        "total",
        "Reason",
        "Old",
    ]
    return pd.DataFrame(rows, columns=columns)


def comparison(result: pd.DataFrame) -> pd.DataFrame:
    """Per level: the players placed, those placed at a slot other than their primary position, those the depth chart's placement
    would put elsewhere, and the sum of the playing values of the lineup."""
    lineup = result[result["Slot"].isin(LINEUP)]
    rows = []
    for level, group in result.groupby("Level", sort=False):
        playing_ = group[group["Slot"].isin([*LINEUP, "SP"])]
        starters = lineup[lineup["Level"] == level]
        rows.append(
            (level, len(group), int((starters["Slot"] != starters["Primary"]).sum()),
             int(((starters["Old"] != "") & (starters["Old"] != starters["Slot"])).sum()), float(playing_["total"].sum()))
        )  # fmt: skip
    return pd.DataFrame(rows, columns=["Level", "Players", "Off primary", "Differs from chart", "Playing value"])
