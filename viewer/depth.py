"""Data of the organization depth chart screen: pure functions over the outlook frame, no UI.

The chart is built the way a manager would build a team. Each pass picks the best set of players for the eight fielding positions
at once, one player per position, so that the total value is as large as possible (an assignment problem); a player goes where
he adds the most to the team given who else is there: the second-best centre fielder who is better at right field than anybody
else plays right field, and the centre fielder spot goes to someone else. The next pass does the same with the players left over,
and so on: pass 1 is the starting team, pass 2 the first backups. Pitchers are assigned the same way to starting and relief
pitcher. A player is only placed where the files rank him (his ratings reach the position's minimum) and appears once.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from viewer import queries

FIELD_SLOTS = ["LF", "CF", "RF", "3B", "SS", "2B", "1B", "C", "SP", "RP"]  # the positions of the game's depth chart
CHART_SLOTS = [*FIELD_SLOTS[:-2], "DH", *FIELD_SLOTS[-2:]]  # the chart of the viewer also has the designated hitter
HITTER_SLOTS = [slot for slot in CHART_SLOTS if slot not in ("SP", "RP")]
PITCHER_SLOTS = {"SP": 5, "RP": 5}  # places per pass, so two passes give the ten of each box
DEPTH_HITTERS = 5
DEPTH_PITCHERS = 10
WHOLE, MINORS = "Entire organization", "Minor leagues"
SHOW = (WHOLE, MINORS)  # and, after these, the level (team) codes of show_options
LEVEL_ORDER = ["AAA", "AA", "A+", "A", "A-", "R+", "R", "Rookie", "INT", "WL"]  # the levels of the files, top to bottom
LEVEL_LABELS = {"INT": "International Complex", "MLB": "MLB roster"}
RANKED_BY = ("Current", "Ceiling (10 years)", "Floor (10 years)")
HANDS = {1: "R", 2: "L"}  # the throwing hand codes of the files
SIDE_LABELS = {"right": "vs RHP", "left": "vs LHP"}  # the lineups against a right- or left-handed starting pitcher
SIDE_CHOICES = ("Overall", *SIDE_LABELS.values())


def _tidy(name: str) -> str:
    """Some files hold the names of the amateur classes in capitals."""
    return name.title() if name.isupper() else name


def show_label(show: str) -> str:
    return LEVEL_LABELS.get(show, show)


def show_options(frame: pd.DataFrame, org: str) -> list[str]:
    """The parts of an organization the chart can show: all of it, its minor leagues, and each level (team) it has players at."""
    levels = set(queries.select_pool(frame, "org", org)["team_level"]) - {"MLB", "FA", "other"}
    known = [level for level in LEVEL_ORDER if level in levels]
    return [
        *SHOW,
        "MLB",
        *known,
        *sorted(levels - set(known)),
    ]  # MLB: the roster alone, as the acquisitions page sees it


def limit_flags(rows: pd.DataFrame, limits: dict) -> list[str]:
    """For each player the limits of his level he exceeds: "" or a short text such as "age 25 > 24; pro years 6 > 5"."""
    flags = []
    for level, age, years in zip(
        rows["team_level"], rows["Age"], rows.get("PROY", pd.Series(np.nan, index=rows.index))
    ):
        limit = limits.get(level)
        found = []
        if limit and limit.max_age and age > limit.max_age:
            found.append(f"age {age:g} > {limit.max_age}")
        if limit and limit.max_pro_years and pd.notna(years) and years > limit.max_pro_years:
            found.append(f"pro years {years:g} > {limit.max_pro_years}")
        flags.append("; ".join(found))
    return flags


def ranking_column(frame: pd.DataFrame, ranked_by: str, side: str | None = None) -> str:
    """The column a ranking uses: the value now, or the 90th / 10th percentile of his best season within the next 10 (the career
    view; the 10-season horizon for outlook files made before it). With `side` ("right" or "left": the pitcher faced) the value
    now against that kind of pitcher; the projections have no split."""
    if ranked_by not in RANKED_BY:
        raise ValueError(f"unknown ranking {ranked_by!r}; known: {RANKED_BY}")
    if side is not None:
        if side not in SIDE_LABELS:
            raise ValueError(f"unknown side {side!r}; known: {tuple(SIDE_LABELS)}")
        if ranked_by != "Current":
            raise ValueError("the lineups against right- and left-handed pitchers are ranked by the current value only")
        if f"value_vs_{side}" not in frame.columns:
            raise ValueError("the predicted-values file of the season has no batting splits: run update_all.py")
        return f"value_vs_{side}"
    if ranked_by == "Current":
        return "value_now"
    source = "ceiling" if ranked_by.startswith("Ceiling") else "floor"
    for horizon in (queries.CAREER, 10):
        if f"{source}_h{horizon}" in frame.columns:
            return f"{source}_h{horizon}"
    raise ValueError(f"the outlook files have no {source} for 10 seasons")


NOT_ALLOWED = -1e6  # a player is never put where the files do not rank him


PREFERENCE = (
    0.05  # runs: the small bonus for playing the position he is listed at, which breaks exact ties between assignments
)


def assign_pass(values: pd.DataFrame, places: dict[str, int], listed: pd.Series | None = None) -> list[tuple[int, str]]:
    """The best way to fill the places once: `values` has a row per player and a column per slot (NaN: not ranked there);
    `places` the number of players each slot takes. Returns (player ID, slot) pairs of the largest total value. `listed` (ID ->
    the position he is listed at) gives a player a bonus of PREFERENCE runs at that position: two assignments with the same total
    value (shortstop and second baseman swapped, say) are then decided by who plays where, not by rounding."""
    columns = [slot for slot, count in places.items() for _ in range(count)]
    if values.empty or not columns:
        return []
    matrix = values[columns].fillna(NOT_ALLOWED).to_numpy(dtype=float)
    if listed is not None:
        home = listed.reindex(values.index).to_numpy()
        matrix = matrix + PREFERENCE * (home[:, None] == np.array(columns)[None, :]) * (matrix > NOT_ALLOWED / 2)
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    return [(int(values.index[r]), columns[c]) for r, c in zip(rows, cols) if matrix[r, c] > NOT_ALLOWED / 2]


def fill(pool: pd.DataFrame, column: str, places: dict[str, int], depth: int) -> dict[str, list[int]]:
    """Slot -> player IDs in depth order: `depth` passes of assign_pass, each over the players the earlier passes left."""
    values = pool[pool["slot"].isin(places)].pivot_table(index="ID", columns="slot", values=column, aggfunc="max")
    values = values.reindex(columns=list(places))
    listed = pool.drop_duplicates("ID").set_index("ID")["LPOS"] if "LPOS" in pool.columns else None
    order: dict[str, list[int]] = {slot: [] for slot in places}
    for _ in range(depth):
        picked = assign_pass(values, places, listed)
        if not picked:
            break
        for player, slot in sorted(
            picked, key=lambda pair: -values.loc[pair[0], pair[1]]
        ):  # best first within the pass
            order[slot].append(player)
        values = values.drop(index=[player for player, _ in picked])
    return order


def _pool(frame: pd.DataFrame, org: str, show: str) -> pd.DataFrame:
    """The chart slots of the players of the part of the organization shown."""
    if show not in show_options(frame, org):
        raise ValueError(f"unknown choice {show!r}; known: {show_options(frame, org)}")
    pool = queries.select_pool(frame, "org", org)
    pool = pool[pool["slot"].isin(CHART_SLOTS)]
    if show == MINORS:
        pool = pool[pool["level_name"] != "MLB"]
    elif show != WHOLE:
        pool = pool[pool["team_level"] == show]
    return pool


def _chart(pool: pd.DataFrame, order: dict[str, list[int]], columns: dict[str, str], limits: dict | None) -> dict:
    """Slot -> its table (ID, Player, Lev, Value, Flag) for the player IDs of `order`; `columns`: the value column per slot."""
    chart = {}
    for slot in CHART_SLOTS:
        rows = pool[pool["slot"] == slot].drop_duplicates("ID").set_index("ID").reindex(order[slot])
        names = (rows["First Name"].astype(str) + " " + rows["Last Name"].astype(str)).map(_tidy)
        if slot in ("SP", "RP"):
            names = rows["T"].map(HANDS).fillna("").add("HP ").where(rows["T"].isin(HANDS), "") + names
        chart[slot] = pd.DataFrame(
            {
                "ID": rows.index.to_numpy(),
                "Player": names.to_numpy(),
                "Lev": rows["team_level"].to_numpy(),
                "Value": rows[columns[slot]].to_numpy(),
                "Flag": limit_flags(rows, limits or {}),
            }
        )
    return chart


def depth_chart(
    frame: pd.DataFrame,
    org: str,
    show: str = WHOLE,
    ranked_by: str = RANKED_BY[0],
    limits: dict | None = None,
    side: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Slot -> its players in depth order (the starter first); columns ID, Player, Lev, Value (the ranking value at the slot) and
    Flag (the limits of his level he exceeds; `limits` is the `levels` of the save's config: level -> LevelLimits).

    `show`: the whole organization, its minor leagues, or one level (team) of show_options. `side`: "right" or "left" for the
    lineup against a right- or left-handed starting pitcher (the hitters' values against that side, ranked by the current value
    only); the pitchers are one list whatever the side.
    """
    column = ranking_column(frame, ranked_by, side)
    pitching = ranking_column(frame, ranked_by)
    pool = _pool(frame, org, show)
    pitcher = pool["slot"].isin(PITCHER_SLOTS)
    hitting = pool[~pitcher].dropna(subset=[column])
    pool = pd.concat([hitting, pool[pitcher].dropna(subset=[pitching])])
    hitters = fill(pool, column, dict.fromkeys(HITTER_SLOTS, 1), DEPTH_HITTERS)
    placed = {player for ids in hitters.values() for player in ids}
    pitchers = fill(pool[~pool["ID"].isin(placed)], pitching, PITCHER_SLOTS, DEPTH_PITCHERS // PITCHER_SLOTS["SP"])
    columns = {slot: pitching if slot in PITCHER_SLOTS else column for slot in CHART_SLOTS}
    return _chart(pool, hitters | pitchers, columns, limits)
