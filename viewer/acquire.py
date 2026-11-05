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
YEARS = 3  # the years looked at for a contract and the pipeline: now, in one year, in two
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
    min_budget: float = (
        10_000_000  # the budget space the priority list leaves untouched (draft, international signings ...)
    )
    multiyear_demand: float = 8_000_000  # a free agent asking at least this much wants more than a year
    multiyear_years: int = 3
    block_weight: float = 0.5  # what a year of a ready prospect kept out of the lineup by a signing counts against it
    non_need_factor: float = 0.5  # an upgrade of a place that is not a need counts this much
    # boost of the expected impact for a very / extremely popular player: local popularity is what counts (the local market,
    # fan interest), national popularity (awards, all-star votes) counts for much less
    popularity_local: tuple = (("vpop", 0.10), ("expop", 0.20))
    popularity_national: tuple = (("vpop", 0.02), ("expop", 0.04))
    sign_chance: tuple = (
        ("Very Easy", 0.95), ("Easy", 0.85), ("Normal", 0.7), ("Hard", 0.45), ("Extremely Hard", 0.2), ("Impossible", 0.0),
    )  # fmt: skip

    def chance(self, sign) -> float:
        return dict(self.sign_chance).get(str(sign), 1.0)

    def weight(self, slot: str) -> float:
        return dict(self.weights).get(kind_of(slot), 1.0)


def load_settings(path: Path = ACQUISITION_CONFIG) -> Settings:
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    values = dict(raw.get("rules", {}))
    if "hard_to_sign" in values:
        values["hard_to_sign"] = tuple(values["hard_to_sign"])
    if "weights" in raw:
        values["weights"] = tuple(sorted(raw["weights"].items()))
    for kind in ("local", "national"):
        if kind in raw.get("popularity", {}):
            values[f"popularity_{kind}"] = tuple((_pop_key(k), v) for k, v in raw["popularity"][kind].items())
    if "signing" in raw:
        values["sign_chance"] = tuple(raw["signing"].items())
    unknown = set(values) - set(Settings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
    return replace(Settings(), **values)


POPULARITY = ("unkn", "insig", "fair", "known", "pop", "vpop", "expop")  # the game's levels, lowest first
POPULARITY_LABELS = {"vpop": "V.Pop", "expop": "Ex.Pop", "pop": "Pop"}


def _pop_key(text) -> str:
    """A popularity text of the export ("V.Pop", "Ex.Pop.", "Unkn.") as a key of POPULARITY."""
    return str(text).lower().replace(".", "").replace(" ", "")


def popularity(rows: pd.DataFrame) -> pd.DataFrame:
    """A player's local and national popularity as keys of POPULARITY (unknown for an export without the columns)."""
    return pd.DataFrame(
        {
            kind: (
                rows[column].map(_pop_key).where(lambda keys: keys.isin(POPULARITY), "unkn")
                if column in rows
                else "unkn"
            )
            for kind, column in (("local", "Loc. Pop."), ("national", "Nat. Pop."))
        },
        index=rows.index,
    )


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
    if "park_effect" in pool:  # the park-adjusted view: what the park changes in the starter's value
        effect = pool.drop_duplicates(["ID", "slot"]).set_index(["ID", "slot"])["park_effect"]
        table["park"] = [effect.get((i, s), np.nan) for i, s in zip(table["ID"], table["slot"])]
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
    mine["Listed"] = mine["ID"].map(people["LPOS"]) if "LPOS" in people else ""  # the position he is listed at
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
    (his team or level), Deal (a free agent who asks nothing signs a minor league deal), Cost (dollars for a year), Sign,
    Years (contract years left), value. Internal: the organization's players who are not starters; Waivers: players of other
    organizations on waivers; Free agent: the free agents (minus those hard to sign by the settings).
    """
    members = ~frame["org"].isin(["FA", "nan", ""])
    taken = set(starters.loc[starters["org"] == str(org), "ID"])
    waiv = frame["WAIV"].astype(str).eq("Yes") if "WAIV" in frame else pd.Series(False, index=frame.index)
    internal = (frame["org"] == str(org)) & ~frame["ID"].isin(taken)
    waivers = members & (frame["org"] != str(org)) & waiv
    free = frame["is_free_agent"] & ~waiv
    rows = frame[internal | waivers | free].dropna(subset=[column]).copy()
    rows = rows[rows["slot"].isin(depth.CHART_SLOTS)]
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
    # a demand of 0 is a player who signs a minor league deal (with a major league option): cheap
    rows["Deal"] = np.where(fa & ~(demand > 0), "Minor league", np.where(fa, "MLB", ""))
    rows["Cost"] = cost
    rows["Sign"] = rows["Sign"] if "Sign" in rows else ""
    rows["Years"] = rows["YL"] if "YL" in rows else ""
    rows = rows[~(fa & rows["Sign"].isin(settings.hard_to_sign))]
    rows["Player"] = rows.apply(_name, axis=1)
    rows[["PopLocal", "PopNational"]] = popularity(rows)
    for n in range(1, YEARS):
        rows[f"y{n}"] = _ahead(rows, n, column)
    keep = [
        "ID",
        "slot",
        "Player",
        "Age",
        "Source",
        "From",
        "Deal",
        "Cost",
        "Sign",
        "Years",
        "PopLocal",
        "PopNational",
        column,
    ]
    keep += [f"y{n}" for n in range(1, YEARS)]
    if "park_effect" in rows:
        keep.append("park_effect")
    return rows[keep].rename(columns={column: "value", "park_effect": "park"}).reset_index(drop=True)


def _ahead(rows: pd.DataFrame, years: int, column: str) -> pd.Series:
    """The value `years` ahead on the scale of `column` (the park adjusted value keeps its park shift); without the horizon
    columns the value now."""
    ahead = f"value_h{years}"
    if ahead not in rows:
        return rows[column]
    return (rows[ahead] + (rows[column] - rows["value_now"])).fillna(rows[column])


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


def _curves(frame: pd.DataFrame, column: str) -> dict:
    """(ID, slot) -> the value now, in one year, in two years (the last known value when a horizon is missing)."""
    people = frame.drop_duplicates(["ID", "slot"]).copy()
    for n in range(1, YEARS):
        people[f"y{n}"] = _ahead(people, n, column)
    people["y0"] = people[column]
    values = people[[f"y{n}" for n in range(YEARS)]].to_numpy(dtype=float)
    return {(i, s): v for i, s, v in zip(people["ID"], people["slot"], values)}


def _contract_years(row: pd.Series, settings: Settings) -> int:
    """How long the deal runs: a free agent who asks a lot wants several years, a claimed player keeps his contract, an own
    player stays."""
    if row["Source"] == FREE_AGENT:
        wants = row["Deal"] == "MLB" and row["Cost"] >= settings.multiyear_demand
        return min(settings.multiyear_years, YEARS) if wants else 1
    if row["Source"] == WAIVERS:
        left = pd.to_numeric(pd.Series([row["Years"]]), errors="coerce").iloc[0]
        return int(min(max(left if pd.notna(left) else 1, 1), YEARS))
    return YEARS


def _pipeline(internal: pd.DataFrame, slot: str, target: float) -> tuple[np.ndarray, int | None, str]:
    """What the organization's own players (not starters) at a slot will be worth in one and two years: the best value of each
    year (index 0 is not counted: those players are candidates now), the first year one of them reaches `target` and his name.
    """
    mine = internal[internal["slot"] == slot]
    curve = np.full(YEARS, -np.inf)
    for n in range(1, YEARS):
        if len(mine) and mine[f"y{n}"].notna().any():
            curve[n] = mine[f"y{n}"].max()
    for n in range(1, YEARS):
        if curve[n] >= target:
            best = mine.loc[mine[f"y{n}"].idxmax()]
            return curve, n, f"{best['Player']} ({best['From']}) in {n} yr"
    return curve, None, ""


def priority_list(
    frame: pd.DataFrame,
    needs: pd.DataFrame,
    pool: pd.DataFrame,
    budget: float,
    settings: Settings,
    column: str = "value_now",
) -> pd.DataFrame:
    """The acquisitions in the order to try them: for each place the candidate with the largest expected impact, best first, at
    most one per place and each candidate once; rows with Place, Player, Source, Cost (a year), Years, Commit, Impact (runs
    over the contract, against the starter and what the organization's own prospects will be), Chance (he signs), Expected,
    PopLocal / PopNational (his popularity: a very / extremely popular player gets a boost because he draws fans, mostly
    the local one), Context (the
    pipeline at the place), Verdict, Spend (Within the budget space above `min_budget`, or Over) and Fallbacks.

    A year in which a signing keeps a ready prospect out of the lineup counts against it (`block_weight`); an upgrade of a place
    that is not a need counts `non_need_factor`. Verdict: Hole (nothing of our own that reaches the target within two years),
    Soon / Bridge (a prospect reaches the target in two / one year: a short deal), Fill (the starter is weak, nobody coming).
    """
    curves = _curves(frame, column)
    internal = pool[pool["Source"] == INTERNAL]
    rows = []
    for _, need in needs.iterrows():
        cands = pool[pool["slot"] == need["slot"]]
        if cands.empty:
            continue
        start = curves.get((need["ID"], need["slot"])) if pd.notna(need["ID"]) else None
        start = np.full(YEARS, need["value"]) if start is None else np.where(np.isnan(start), need["value"], start)
        pipe, ready, who = _pipeline(internal, need["slot"], need["target"])
        base = np.maximum(start, pipe)
        grid = cands[["value", *[f"y{n}" for n in range(1, YEARS)]]].to_numpy(dtype=float)
        gain = np.clip(grid - base, 0, None)
        block = np.where(np.isfinite(pipe), np.clip(pipe - grid, 0, None), 0.0)
        length = np.array([_contract_years(r, settings) for _, r in cands.iterrows()])
        inside = np.arange(YEARS)[None, :] < length[:, None]
        impact = ((gain - settings.block_weight * block) * inside).sum(axis=1) * settings.weight(need["place"])
        impact *= 1.0 if need["need"] else settings.non_need_factor
        chance = np.where(cands["Source"] == FREE_AGENT, cands["Sign"].map(settings.chance).fillna(1.0), 1.0)
        verdict = "Hole" if ready is None else ("Bridge" if ready == 1 else "Soon")
        if ready is None and not need["need"]:
            verdict = "Fill"
        out = cands[
            ["ID", "slot", "Player", "Age", "Source", "From", "Deal", "Cost", "Sign", "PopLocal", "PopNational"]
        ].copy()
        out["Place"] = need["place"]
        out["Years"] = length
        out["Commit"] = out["Cost"] * length
        out["Impact"] = impact
        out["PerYear"] = impact / length
        out["Chance"] = chance
        boost = (
            1
            + cands["PopLocal"].map(dict(settings.popularity_local)).fillna(0.0).to_numpy()
            + cands["PopNational"].map(dict(settings.popularity_national)).fillna(0.0).to_numpy()
        )
        out["Boost"] = boost
        out["Expected"] = impact * chance * boost
        out["Context"] = who
        out["Verdict"] = verdict
        out["Need"] = bool(need["need"])
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    every = pd.concat(rows, ignore_index=True)
    room = budget if pd.notna(budget) else np.inf
    every = every[(every["Expected"] >= settings.min_gain) & (every["Cost"] <= room - settings.min_budget)].sort_values(
        "Expected", ascending=False, kind="stable"
    )
    chosen, used_places, used_people = [], set(), set()
    for index, row in every.iterrows():
        if row["Place"] in used_places or row["ID"] in used_people:
            continue
        chosen.append(index)
        used_places.add(row["Place"])
        used_people.add(row["ID"])
    top = every.loc[chosen].copy()
    top["Fallbacks"] = [
        ", ".join(
            f"{r['Player']} ({r['Expected']:.1f})"
            for _, r in every[(every["Place"] == row["Place"]) & ~every["ID"].isin(used_people)].head(2).iterrows()
        )
        for _, row in top.iterrows()
    ]
    spent, status = 0.0, []
    for cost in top[
        "Cost"
    ]:  # in order; a deal that would take the budget space under the minimum is skipped, cheaper ones after it still can
        fits = spent + cost <= room - settings.min_budget
        status.append("Within" if fits else "Over")
        spent += cost if fits else 0.0
    top["Spend"] = status
    return top.reset_index(drop=True)
