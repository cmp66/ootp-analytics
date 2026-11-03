"""Data of the organization depth chart screen: pure functions over the outlook frame, no UI.

The chart is built the way a manager would build a team. Each pass picks the best set of players for the eight fielding positions
at once, one player per position, so that the total value is as large as possible (an assignment problem); a player goes where
he adds the most to the team given who else is there: the second-best centre fielder who is better at right field than anybody
else plays right field, and the centre fielder spot goes to someone else. The next pass does the same with the players left over,
and so on: pass 1 is the starting team, pass 2 the first backups. Pitchers are assigned the same way to starting and relief
pitcher. A player is only placed where the files rank him (his ratings reach the position's minimum) and appears once.
"""

import pandas as pd
from scipy.optimize import linear_sum_assignment

from viewer import queries

CHART_SLOTS = ["LF", "CF", "RF", "3B", "SS", "2B", "1B", "C", "SP", "RP"]
HITTER_SLOTS = [slot for slot in CHART_SLOTS if slot not in ("SP", "RP")]
PITCHER_SLOTS = {"SP": 5, "RP": 5}  # places per pass, so two passes give the ten of each box
DEPTH_HITTERS = 5
DEPTH_PITCHERS = 10
SHOW = ("Entire organization", "Minor leagues")
RANKED_BY = ("Current", "Ceiling (10 years)", "Floor (10 years)")
HANDS = {1: "R", 2: "L"}  # the throwing hand codes of the files


def _tidy(name: str) -> str:
    """Some files hold the names of the amateur classes in capitals."""
    return name.title() if name.isupper() else name


def ranking_column(frame: pd.DataFrame, ranked_by: str) -> str:
    """The column a ranking uses: the value now, or the 90th / 10th percentile of his best season within the next 10 (the career
    view; the 10-season horizon for outlook files made before it)."""
    if ranked_by not in RANKED_BY:
        raise ValueError(f"unknown ranking {ranked_by!r}; known: {RANKED_BY}")
    if ranked_by == "Current":
        return "value_now"
    source = "ceiling" if ranked_by.startswith("Ceiling") else "floor"
    for horizon in (queries.CAREER, 10):
        if f"{source}_h{horizon}" in frame.columns:
            return f"{source}_h{horizon}"
    raise ValueError(f"the outlook files have no {source} for 10 seasons")


NOT_ALLOWED = -1e6  # a player is never put where the files do not rank him


def assign_pass(values: pd.DataFrame, places: dict[str, int]) -> list[tuple[int, str]]:
    """The best way to fill the places once: `values` has a row per player and a column per slot (NaN: not ranked there);
    `places` the number of players each slot takes. Returns (player ID, slot) pairs of the largest total value."""
    columns = [slot for slot, count in places.items() for _ in range(count)]
    if values.empty or not columns:
        return []
    matrix = values[columns].fillna(NOT_ALLOWED).to_numpy(dtype=float)
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    return [(int(values.index[r]), columns[c]) for r, c in zip(rows, cols) if matrix[r, c] > NOT_ALLOWED / 2]


def fill(pool: pd.DataFrame, column: str, places: dict[str, int], depth: int) -> dict[str, list[int]]:
    """Slot -> player IDs in depth order: `depth` passes of assign_pass, each over the players the earlier passes left."""
    values = pool[pool["slot"].isin(places)].pivot_table(index="ID", columns="slot", values=column, aggfunc="max")
    values = values.reindex(columns=list(places))
    order: dict[str, list[int]] = {slot: [] for slot in places}
    for _ in range(depth):
        picked = assign_pass(values, places)
        if not picked:
            break
        for player, slot in sorted(
            picked, key=lambda pair: -values.loc[pair[0], pair[1]]
        ):  # best first within the pass
            order[slot].append(player)
        values = values.drop(index=[player for player, _ in picked])
    return order


def depth_chart(
    frame: pd.DataFrame, org: str, show: str = SHOW[0], ranked_by: str = RANKED_BY[0]
) -> dict[str, pd.DataFrame]:
    """Slot -> its players in depth order (the starter first); columns ID, Player, Lev, Value (the ranking value at the slot)."""
    if show not in SHOW:
        raise ValueError(f"unknown choice {show!r}; known: {SHOW}")
    column = ranking_column(frame, ranked_by)
    pool = queries.select_pool(frame, "org", org)
    pool = pool[pool["slot"].isin(CHART_SLOTS)].dropna(subset=[column])
    if show == SHOW[1]:
        pool = pool[pool["level_name"] != "MLB"]
    hitters = fill(pool, column, dict.fromkeys(HITTER_SLOTS, 1), DEPTH_HITTERS)
    placed = {player for ids in hitters.values() for player in ids}
    pitchers = fill(pool[~pool["ID"].isin(placed)], column, PITCHER_SLOTS, DEPTH_PITCHERS // PITCHER_SLOTS["SP"])
    order = hitters | pitchers
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
                "Lev": rows["level_name"].to_numpy(),
                "Value": rows[column].to_numpy(),
            }
        )
    return chart
