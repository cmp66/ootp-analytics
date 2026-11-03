"""Data of the minor league management page: pure functions over the outlook frame and the ratings, no UI.

For every minor leaguer of an organization: where his key ratings stand among all the players of his level (the way the game shows
ratings relative to a level), where they would stand at the next level up, how old he is for the level, and from that a move: up,
down or must move (over the level's limits), with the level he goes to and why. See config/minors.toml for the rules.

The focus is the development of the better players while they can still develop: a player is a top prospect, a prospect or depth
by where his ceiling (the 90th percentile of his best season within ten) stands among all minor leaguers of his position in the
league, and his development window is what is left until the age at which players of his kind stop improving (measured in the
training leagues, config/minors.toml). Depth players and players past their window are filler: they are only listed on request.
"""

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from util.config import LevelLimits
from viewer import depth

MINORS_CONFIG = Path(__file__).resolve().parent.parent / "config" / "minors.toml"
TOP = "MLB"  # the level above the ladder
LABELS = {  # short column titles of the key ratings
    "CON": "Con", "GAP": "Gap", "POW": "Pow", "EYE": "Eye", "K's": "AvK", "BABIP": "BABIP",
    "STU": "Stu", "MOV": "Mov", "CON.1": "Ctl",
}  # fmt: skip
UP, DOWN, MUST = "Up", "Down", "Must move"
TOP_PROSPECT, PROSPECT, PAST, DEPTH = "Top prospect", "Prospect", "Past window", "Depth"


@dataclass(frozen=True)
class Rules:
    hitters: tuple = ("CON", "GAP", "POW", "EYE", "K's", "BABIP")
    pitchers: tuple = ("STU", "MOV", "CON.1")
    green: float = 0.70
    red: float = 0.25
    average: float = 0.50
    min_green_hitters: int = 4
    min_green_pitchers: int = 2
    next_floor: float = 0.35
    old_for_level: float = 0.75
    prospect: float = 0.80
    top_prospect: float = 0.95
    develop_hitters: tuple = (
        ("OOTP26", 25),
        ("OOTP27", 25),
    )  # the last age at which ratings still grow, by game version
    develop_pitchers: tuple = (("OOTP26", 26), ("OOTP27", 26))

    def keys(self, pitcher: bool) -> tuple:
        return self.pitchers if pitcher else self.hitters

    def min_green(self, pitcher: bool) -> int:
        return self.min_green_pitchers if pitcher else self.min_green_hitters

    def develops_through(self, version: str, pitcher: bool) -> int:
        """The last age at which players of this kind still improve in this game version."""
        table = dict(self.develop_pitchers if pitcher else self.develop_hitters)
        if version not in table:
            raise ValueError(f"config/minors.toml has no development age for {version} (known: {sorted(table)})")
        return table[version]


def load_rules(path: Path = MINORS_CONFIG) -> Rules:
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    values = dict(raw.get("rules", {}))
    for kind in ("hitters", "pitchers"):
        if kind in raw.get("ratings", {}):
            values[kind] = tuple(raw["ratings"][kind])
        if kind in raw.get("development", {}):
            values[f"develop_{kind}"] = tuple(sorted(raw["development"][kind].items()))
    unknown = set(values) - set(Rules.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown settings {sorted(unknown)}")
    return replace(Rules(), **values)


def ladder(levels: dict[str, LevelLimits]) -> list[str]:
    """The levels of the promotion ladder, top first: the order of the level tables of the save, minus those off the ladder."""
    return [level for level, limits in levels.items() if limits.ladder]


def percentile(values: pd.Series, peers: np.ndarray) -> pd.Series:
    """Where each value stands among the peers: the share of them below it, ties counting half (0 to 1)."""
    peers = np.sort(peers[~np.isnan(peers)])
    if len(peers) == 0:
        return pd.Series(np.nan, index=values.index)
    v = values.to_numpy(dtype=float)
    return pd.Series(
        (np.searchsorted(peers, v, "left") + np.searchsorted(peers, v, "right")) / 2 / len(peers), index=values.index
    )


def prospect_info(frame: pd.DataFrame) -> pd.DataFrame:
    """Per minor leaguer of the league (ID): the ceiling and floor of his best season within ten at the position where his ceiling
    stands highest, where they stand (percentile among the minor leaguers of his kind in the league) and his chance of ever being
    a top-50 player at it."""
    ceiling = depth.ranking_column(frame, "Ceiling (10 years)")
    floor = depth.ranking_column(frame, "Floor (10 years)")
    chance = ceiling.replace("ceiling", "p_top50")
    rows = frame[
        frame["slot"].isin(depth.CHART_SLOTS) & ~frame["org"].isin(["FA", "nan", ""]) & (frame["team_level"] != TOP)
    ]
    rows = rows.assign(
        ceiling_pct=rows.groupby("slot")[ceiling].rank(pct=True), floor_pct=rows.groupby("slot")[floor].rank(pct=True)
    )
    best = rows.sort_values("ceiling_pct", ascending=False, kind="stable").drop_duplicates("ID").copy()
    # the highest of his positions' percentiles is high for everybody: rank those among all the minor leaguers of his kind, so
    # that 0.8 means the better fifth of them (hitters and pitchers apart)
    best["pitcher"] = best["slot"].isin(["SP", "RP"])
    best["ceiling_pct"] = best.groupby("pitcher")["ceiling_pct"].rank(pct=True)
    best["floor_pct"] = best.groupby("pitcher")["floor_pct"].rank(pct=True)
    info = pd.DataFrame(
        {"ID": best["ID"], "Ceil": best[ceiling], "Floor": best[floor], "ceiling_pct": best["ceiling_pct"],
         "floor_pct": best["floor_pct"], "Top50": best[chance] * 100 if chance in best else np.nan}
    )  # fmt: skip
    return info.reset_index(drop=True)


def league_players(frame: pd.DataFrame, ratings: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """One row per player who belongs to an organization: ID, org, team_level, Age, PROY, pitcher, his best slot (by value now),
    value now, name, and the key ratings of both kinds."""
    rows = frame[frame["slot"].isin(depth.CHART_SLOTS)].copy()
    rows["standing"] = rows.groupby("slot")["value_now"].rank(
        pct=True
    )  # his position is where he stands highest in the league
    best = rows.sort_values("standing", ascending=False, kind="stable").drop_duplicates("ID")
    best = best[~best["org"].isin(["FA", "nan", ""])]
    columns = ["ID", "org", "team_level", "Age", "slot", "value_now", "First Name", "Last Name"] + (
        ["PROY"] if "PROY" in best else []
    )
    out = best[columns].copy()
    if "PROY" not in out:
        out["PROY"] = np.nan
    out["pitcher"] = out["slot"].isin(["SP", "RP"])
    out = out.merge(prospect_info(frame), on="ID", how="left")
    keys = [k for k in dict.fromkeys([*rules.hitters, *rules.pitchers]) if k in ratings.columns]
    out = out.merge(ratings[["ID", *keys]], on="ID", how="left")
    for key in keys:
        out[key] = pd.to_numeric(out[key], errors="coerce").where(lambda s: s > 0)
    return out.reset_index(drop=True)


def manage(
    frame: pd.DataFrame,
    ratings: pd.DataFrame,
    org: str,
    levels: dict[str, LevelLimits],
    rules: Rules | None = None,
    version: str = "OOTP27",
) -> pd.DataFrame:
    """The minor leaguers of `org` on the ladder: their ratings relative to the level and the move suggested.

    Columns: ID, Player, Pos, Age, Pro, Level, pitcher, the key ratings as percentiles at the level (`pct_<rating>`) and at the
    next level up (`next_<rating>`), green (how many are green), mean, next_mean (mean of the percentiles at the next level),
    age_pct, Value, Ceil, Floor, Top50 (the chance in % of ever being top 50 at the position), ceiling_pct, Tier (top prospect,
    prospect or depth, by the ceiling_pct among the league's minor leaguers), Window (seasons of development left until the age
    at which this kind of player stops improving in this game `version`), focus (a prospect with a window left), Move ("", Up,
    Down or Must move), To (the level), Reason, Blocked (why the limits do not allow it, else ""). Players of organizations
    other than `org` are the peers only.
    """
    rules = rules or Rules()
    rungs = ladder(levels)
    players = league_players(frame, ratings, rules)
    mine = players[(players["org"] == str(org)) & players["team_level"].isin(rungs)].copy()
    rows = []
    for pitcher in (False, True):
        keys = [k for k in rules.keys(pitcher) if k in players.columns]
        for level in rungs:
            group = mine[(mine["pitcher"] == pitcher) & (mine["team_level"] == level)]
            if group.empty or not keys:
                continue
            peers = players[(players["pitcher"] == pitcher) & (players["team_level"] == level)]
            above = rungs[rungs.index(level) - 1] if rungs.index(level) > 0 else TOP
            above_peers = players[(players["pitcher"] == pitcher) & (players["team_level"] == above)]
            below = rungs[rungs.index(level) + 1] if rungs.index(level) + 1 < len(rungs) else None
            out = group.copy()
            for key in keys:
                out[f"pct_{key}"] = percentile(group[key], peers[key].to_numpy(dtype=float))
                out[f"next_{key}"] = percentile(group[key], above_peers[key].to_numpy(dtype=float))
            out["green"] = (out[[f"pct_{k}" for k in keys]] >= rules.green).sum(axis=1)
            out["mean"] = out[[f"pct_{k}" for k in keys]].mean(axis=1)
            out["next_mean"] = out[[f"next_{k}" for k in keys]].mean(axis=1)
            out["age_pct"] = percentile(group["Age"], peers["Age"].to_numpy(dtype=float))
            out["above"], out["below"] = above, below
            out["keys"] = len(keys)
            rows.append(out)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    last_age = table["pitcher"].map(lambda pitcher: rules.develops_through(version, bool(pitcher)))
    table["Window"] = (last_age - table["Age"] + 1).clip(lower=0).astype(int)
    table["Tier"] = np.where(
        table["ceiling_pct"] >= rules.top_prospect,
        TOP_PROSPECT,
        np.where(table["ceiling_pct"] >= rules.prospect, PROSPECT, DEPTH),
    )
    table["Tier"] = np.where(
        (table["Window"] == 0) & (table["Tier"] != DEPTH), PAST, table["Tier"]
    )  # nothing left to develop
    table["focus"] = table["Tier"].isin([TOP_PROSPECT, PROSPECT])
    decided = table.apply(lambda r: _decide(r, levels, rules), axis=1, result_type="expand")
    table[["Move", "To", "Reason", "Blocked"]] = decided
    table["Player"] = (table["First Name"].astype(str) + " " + table["Last Name"].astype(str)).map(depth._tidy)
    table = table.rename(columns={"slot": "Pos", "PROY": "Pro", "team_level": "Level", "value_now": "Value"})
    return table.drop(columns=["First Name", "Last Name", "org", "above", "below", "keys"])


def _fits(level: str | None, age, years, levels: dict[str, LevelLimits]) -> str:
    """ "" if a player of this age and pro years may play at the level, else why not."""
    if level is None:
        return "no level below"
    limit = levels.get(level)
    if limit is None:
        return ""
    if limit.max_age and age > limit.max_age:
        return f"age {age:g} > {limit.max_age} at {level}"
    if limit.max_pro_years and pd.notna(years) and years > limit.max_pro_years:
        return f"pro years {years:g} > {limit.max_pro_years} at {level}"
    return ""


def _decide(row: pd.Series, levels: dict[str, LevelLimits], rules: Rules) -> tuple[str, str, str, str]:
    """(move, to, reason, blocked) for one player of the table of manage."""
    level, age, years = row["team_level"], row["Age"], row["PROY"]
    over = _fits(level, age, years, levels)
    can_up = _fits(row["above"], age, years, levels) if row["above"] != TOP else ""
    ready = pd.notna(row["next_mean"]) and row["next_mean"] >= rules.next_floor
    needed = rules.min_green(bool(row["pitcher"]))
    if row["focus"]:
        needed = max(
            1, needed - 1
        )  # a prospect with development left needs the challenge more than he needs to dominate
    if over:
        return MUST, row["above"], f"over the limit of the level ({over})", can_up
    if row["green"] >= needed and ready:
        return (
            UP,
            row["above"],
            f"{row['green']} of {row['keys']} key ratings green at the level; mean {row['next_mean']:.0%} at {row['above']}",
            can_up,
        )
    if row["focus"] and row["age_pct"] >= rules.old_for_level and row["mean"] >= rules.average and ready:
        return (
            UP,
            row["above"],
            f"old for the level (older than {row['age_pct']:.0%} of it) with average or better ratings",
            can_up,
        )
    if row["below"] and row["mean"] <= rules.red and row["green"] == 0:
        return (
            DOWN,
            row["below"] or "",
            f"weak for the level (mean percentile {row['mean']:.0%}, no key rating green)",
            _fits(row["below"], age, years, levels),
        )
    return "", "", "", ""
