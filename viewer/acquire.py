"""Data of the roster acquisition page: pure functions over the outlook frame and the finances, no UI.

Where an organization's MLB roster needs an upgrade and who could provide it: its own players who are not starters, players on
waivers and free agents (config/acquisition.toml has the rules). A candidate's gain is his value at the place minus the value of
the starter he would replace; the best set of acquisitions under the budget space is found by an integer program.
"""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from stats.finances import parse_money
from viewer import depth

ACQUISITION_CONFIG = Path(__file__).resolve().parent.parent / "config" / "acquisition.toml"
LINEUP = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
ROTATION = 5
INTERNAL, WAIVERS, FREE_AGENT = "Internal", "Waivers", "Free agent"
SOURCES = (INTERNAL, WAIVERS, FREE_AGENT)


@dataclass(frozen=True)
class Settings:
    need_percentile: float = 0.40
    target_percentile: float = 0.50
    bullpen: int = 7
    min_gain: float = 1.0
    hard_to_sign: tuple = ("Impossible",)
    weights: tuple = (("RP", 0.35),)

    def weight(self, slot: str) -> float:
        return dict(self.weights).get(kind_of(slot), 1.0)


def load_settings(path: Path = ACQUISITION_CONFIG) -> Settings:
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    values = dict(raw.get("rules", {}))
    if "hard_to_sign" in values:
        values["hard_to_sign"] = tuple(values["hard_to_sign"])
    if "weights" in raw:
        values["weights"] = tuple(sorted(raw["weights"].items()))
    unknown = set(values) - set(Settings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
    return replace(Settings(), **values)


def kind_of(place: str) -> str:
    """The slot of a place of the roster: C ... DH, SP (SP1 ... SP5) or RP (RP1 ... RP7)."""
    return place[:2] if place[:2] in ("SP", "RP") and place[2:].isdigit() else place


def places(settings: Settings) -> list[str]:
    return [*LINEUP, *[f"SP{i}" for i in range(1, ROTATION + 1)], *[f"RP{i}" for i in range(1, settings.bullpen + 1)]]


def mlb_players(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["level_name"] == "MLB") & ~frame["org"].isin(["FA", "nan", ""])]


def team_starters(frame: pd.DataFrame, settings: Settings, column: str = "value_now") -> pd.DataFrame:
    """The starters of every organization: org, place (C ... DH, SP1 ... SP5, RP1 ...), ID, slot and value, from its MLB players.

    Hitters: the best lineup, one per position (maximum total value); pitchers: the best five starters, then the best relievers
    of the rest; each in order of value within the place group (SP1 the best)."""
    out = []
    pool = mlb_players(frame).dropna(subset=[column])
    for org, players in pool.groupby("org"):
        hitters = depth.fill(players, column, dict.fromkeys(LINEUP, 1), 1)
        placed = {i for ids in hitters.values() for i in ids}
        rest = players[~players["ID"].isin(placed)]
        pitchers = depth.fill(rest, column, {"SP": ROTATION, "RP": settings.bullpen}, 1)
        for slot, ids in hitters.items():
            out += [(org, slot, i, slot) for i in ids]
        for slot, ids in pitchers.items():
            out += [(org, f"{slot}{n}", i, slot) for n, i in enumerate(ids, 1)]
    table = pd.DataFrame(out, columns=["org", "place", "ID", "slot"])
    value = pool.drop_duplicates(["ID", "slot"]).set_index(["ID", "slot"])[column]
    table["value"] = [value.get((i, s), np.nan) for i, s in zip(table["ID"], table["slot"])]
    return table


def needs(frame: pd.DataFrame, org: str, settings: Settings, column: str = "value_now") -> pd.DataFrame:
    """The starters of `org` with where each stands among the league's starters at the same place group: percentile, the value
    that would be the target, the gap to it and whether the place is a need. Sorted with the biggest (weighted) gaps first.
    """
    starters = team_starters(frame, settings, column)
    starters["group"] = starters["slot"]
    starters["pct"] = starters.groupby("group")["value"].rank(pct=True)
    league = starters.groupby("group")["value"].quantile(settings.target_percentile)
    mine = starters[starters["org"] == str(org)].copy()
    empty = [p for p in places(settings) if p not in set(mine["place"])]  # a place nobody on the roster can play
    worst = starters.groupby("group")["value"].min()
    vacant = pd.DataFrame(
        {
            "org": str(org),
            "place": empty,
            "ID": np.nan,
            "slot": [kind_of(p) for p in empty],
            "group": [kind_of(p) for p in empty],
        }
    )
    vacant["value"] = vacant["group"].map(worst)
    vacant["pct"] = 0.0
    mine = pd.concat([mine, vacant], ignore_index=True)
    mine["target"] = mine["group"].map(league)
    mine["gap"] = (mine["target"] - mine["value"]).clip(lower=0)
    mine["weighted_gap"] = mine["gap"] * mine["place"].map(settings.weight)
    mine["need"] = mine["pct"] <= settings.need_percentile
    people = mlb_players(frame).drop_duplicates("ID").set_index("ID")
    mine["Player"] = mine["ID"].map(lambda i: _name(people.loc[i]) if pd.notna(i) else "(nobody)")
    mine["Age"] = mine["ID"].map(people["Age"])
    order = {p: n for n, p in enumerate(places(settings))}
    return mine.sort_values(
        ["need", "weighted_gap", "place"], ascending=[False, False, True], key=_key(order)
    ).reset_index(drop=True)


def _key(order: dict):
    def key(column: pd.Series) -> pd.Series:
        return column.map(order) if column.name == "place" else column

    return key


def _name(row: pd.Series) -> str:
    return depth._tidy(f"{row['First Name']} {row['Last Name']}")


def minimum_salary(frame: pd.DataFrame) -> float:
    """What the lowest paid MLB player gets: the cost assumed for a free agent who has no demand listed."""
    salary = mlb_players(frame)["SLR"].map(parse_money) if "SLR" in frame else pd.Series(dtype=float)
    salary = salary[salary > 0]
    return float(salary.min()) if len(salary) else 0.0


def candidate_pool(
    frame: pd.DataFrame, org: str, starters: pd.DataFrame, settings: Settings, column: str = "value_now"
) -> pd.DataFrame:
    """Every player who could be an upgrade, with his value at every slot he is ranked at: ID, slot, Player, Age, Source, From
    (his team or level), Cost (dollars for a year), Sign, Years (contract years left), value. Internal: the organization's
    players who are not starters; Waivers: players of other organizations on waivers; Free agent: the free agents (minus those
    who are hard to sign by the settings).
    """
    members = ~frame["org"].isin(["FA", "nan", ""])
    taken = set(starters.loc[starters["org"] == str(org), "ID"])
    waiv = frame["WAIV"].astype(str).eq("Yes") if "WAIV" in frame else pd.Series(False, index=frame.index)
    internal = (frame["org"] == str(org)) & ~frame["ID"].isin(taken)
    waivers = members & (frame["org"] != str(org)) & waiv
    free = frame["is_free_agent"] & ~waiv
    rows = frame[internal | waivers | free].dropna(subset=[column]).copy()
    rows = rows[rows["slot"].isin(depth.CHART_SLOTS + ["DH"])]
    rows["Source"] = np.where(internal[rows.index], INTERNAL, np.where(waivers[rows.index], WAIVERS, FREE_AGENT))
    rows["From"] = np.where(
        rows["Source"] == FREE_AGENT, "FA", np.where(rows["Source"] == INTERNAL, rows["team_level"], rows["org"])
    )
    cost = pd.Series(0.0, index=rows.index)
    salary = rows["SLR"].map(parse_money) if "SLR" in rows else pd.Series(np.nan, index=rows.index)
    demand = rows["DEM"].map(parse_money) if "DEM" in rows else pd.Series(np.nan, index=rows.index)
    cost[rows["Source"] == WAIVERS] = salary[rows["Source"] == WAIVERS].where(salary > 0).fillna(minimum_salary(frame))
    fa = rows["Source"] == FREE_AGENT
    cost[fa] = demand[fa].where(demand[fa] > 0).fillna(minimum_salary(frame))  # no demand listed: a minimum contract
    rows["Cost"] = cost
    rows["Sign"] = rows["Sign"] if "Sign" in rows else ""
    rows["Years"] = rows["YL"] if "YL" in rows else ""
    rows = rows[~(fa & rows["Sign"].isin(settings.hard_to_sign))]
    rows["Player"] = rows.apply(_name, axis=1)
    keep = ["ID", "slot", "Player", "Age", "Source", "From", "Cost", "Sign", "Years", column]
    return rows[keep].rename(columns={column: "value"}).reset_index(drop=True)


def options(pool: pd.DataFrame, need: pd.Series, settings: Settings) -> pd.DataFrame:
    """The candidates for one place of the roster (a row of `needs`): those ranked at its slot who add at least min_gain over
    its starter, best gain first. Gain is weighted (the weight of the place)."""
    rows = pool[pool["slot"] == need["slot"]].copy()
    rows["Gain"] = (rows["value"] - need["value"]) * settings.weight(need["place"])
    rows["Place"] = need["place"]
    rows = rows[rows["Gain"] >= settings.min_gain]
    return rows.sort_values("Gain", ascending=False).reset_index(drop=True)


def best_package(starters: pd.DataFrame, pool: pd.DataFrame, budget: float, settings: Settings) -> pd.DataFrame:
    """The set of acquisitions with the largest total gain whose cost is within `budget`: at most one per place, each candidate
    once (an integer program). Rows of `options` for the chosen pairs, with Place and Gain."""
    pairs = pd.concat([options(pool, row, settings) for _, row in starters.iterrows()], ignore_index=True)
    if pairs.empty:
        return pairs
    n = len(pairs)
    people = pd.factorize(pairs["ID"])[0]
    spots = pd.factorize(pairs["Place"])[0]
    rows = []
    for index in (people, spots):
        matrix = np.zeros((index.max() + 1, n))
        matrix[index, np.arange(n)] = 1
        rows.append(LinearConstraint(matrix, -np.inf, 1))
    rows.append(LinearConstraint(pairs["Cost"].to_numpy(dtype=float)[None, :], -np.inf, max(budget, 0.0)))
    result = milp(-pairs["Gain"].to_numpy(dtype=float), constraints=rows, integrality=np.ones(n), bounds=Bounds(0, 1))
    if not result.success:
        return pairs.iloc[0:0]
    return pairs[result.x > 0.5].sort_values("Gain", ascending=False).reset_index(drop=True)


def budget_space(finances: pd.DataFrame | None, org: str) -> float:
    """The money still available for the season for a team (NaN without finances or when the team is not in them)."""
    if finances is None or "BudgetSpace" not in finances:
        return np.nan
    row = finances.loc[finances["Team"] == str(org), "BudgetSpace"]
    return float(row.iloc[0]) if len(row) else np.nan
