"""Ballpark ratings: load the park factor csv and turn them into per-player effective factors.

A park factor is multiplicative (1.000 is neutral). A player only plays `home_share` of his games in his own
park, so his effective factor is home_share * park + (1 - home_share) * 1.0, averaged over the teams he played for
(weighted by plate appearances / outs). Unknown parks and minor-league players count as neutral.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from util import paths
from util.config import SaveConfig, get_save

logger = logging.getLogger(__name__)

# csv column -> name used here
FACTOR_COLUMNS = {
    "PF AVG": "AVG",
    "AVG L": "AVG_L",
    "AVG R": "AVG_R",
    "PF HR": "HR",
    "HR L": "HR_L",
    "HR R": "HR_R",
    "PF D": "D",
    "PF T": "T",
    "PF": "RUNS",
}
BATTING_FACTORS = ["AVG_L", "AVG_R", "HR_L", "HR_R", "D", "T"]
PITCHING_FACTORS = ["RUNS"]
FACTOR_RANGE = (0.3, 2.5)  # triples factors of 2.0 exist, anything outside this is a typo


@dataclass
class ParkFactors:
    """Effective factors per player ID (batting columns BATTING_FACTORS, pitching columns PITCHING_FACTORS)."""

    batting: pd.DataFrame
    pitching: pd.DataFrame


def ballpark_file(save: SaveConfig, season: int) -> Path | None:
    """The season's own ballpark csv if there is one, else the save-level file, else None (no parks)."""
    if not save.ballparks:
        return None
    seasonal = paths.export_dir(save.name, season) / Path(save.ballparks).name
    if seasonal.exists():
        return seasonal
    shared = paths.save_dir(save.name) / save.ballparks
    if not shared.exists():
        raise FileNotFoundError(
            f"{save.name}: ballparks = {save.ballparks!r} is configured but {shared} does not exist"
        )
    return shared


def load_ballparks(path: Path) -> pd.DataFrame:
    """Park factors indexed by Team Name, with the columns of FACTOR_COLUMNS renamed."""
    df = pd.read_csv(path)
    df["Team Name"] = df["Team Name"].str.strip()
    missing = [c for c in ["Team Name", *FACTOR_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["Team Name"].duplicated().any():
        raise ValueError(f"{path}: team listed twice: {df.loc[df['Team Name'].duplicated(), 'Team Name'].tolist()}")
    df = df.rename(columns=FACTOR_COLUMNS).set_index("Team Name")
    factors = df[list(FACTOR_COLUMNS.values())]
    if factors.isna().any().any():
        raise ValueError(f"{path}: empty park factor values")
    low, high = FACTOR_RANGE
    if ((factors < low) | (factors > high)).any().any():
        raise ValueError(f"{path}: park factors outside {FACTOR_RANGE}")
    return df


def dump_team_parks(league: str, season: int, table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Dump team_id -> "Team Name" in the ballpark table (NaN when there is no park), and a team_id -> label Series."""
    save = get_save(league)
    teams = pd.read_csv(paths.dump_dir(league, save.ratings, season) / "teams.csv", low_memory=False)
    teams = teams[teams["level"] == 1]
    full = teams["name"].where(teams["name"] != teams["nickname"], teams["nickname"]) + np.where(
        teams["name"] != teams["nickname"], " " + teams["nickname"], ""
    )
    full = pd.Series(full.to_numpy(), index=teams["team_id"].to_numpy())
    abbr = pd.Series(teams["abbr"].to_numpy(), index=teams["team_id"].to_numpy())

    def resolve(team_id) -> str | float:
        for key in (full[team_id], abbr[team_id]):
            if key in save.team_names:
                return save.team_names[key]
        return full[team_id] if full[team_id] in table.index else np.nan

    return pd.Series({t: resolve(t) for t in full.index}), full


def dump_stints(league: str, season: int, kind: str, team_parks: pd.Series, team_label: pd.Series) -> pd.DataFrame:
    """One row per player and team stint at the major league level: ID, park, team, weight."""
    save = get_save(league)
    source = paths.dump_dir(league, save.ratings, season) / f"players_career_{kind}_stats.csv"
    weight_columns = ["pa"] if kind == "batting" else ["ip", "ipf"]
    df = pd.read_csv(
        source, usecols=["player_id", "year", "team_id", "level_id", "split_id", *weight_columns], low_memory=False
    )
    df = df[(df["year"] == season) & (df["level_id"] == 1) & (df["split_id"] == 1)]
    weight = df["pa"] if kind == "batting" else df["ip"] * 3 + df["ipf"]
    stints = (
        pd.DataFrame({"ID": df["player_id"], "team_id": df["team_id"], "weight": weight})
        .groupby(["ID", "team_id"], as_index=False)["weight"]
        .sum()
    )
    stints["park"] = stints["team_id"].map(team_parks)
    stints["team"] = stints["team_id"].map(team_label)
    stints["expected"] = True
    return stints[["ID", "park", "team", "weight", "expected"]]


def export_stints(ids: pd.Series, df_ratings: pd.DataFrame, table: pd.DataFrame, team_names: dict) -> pd.DataFrame:
    """Export saves only know a player's current team (ratings TM); only major league players get a park."""
    ratings = df_ratings.reindex(ids.to_numpy())
    major = (ratings["Lev"] == "MLB").to_numpy()
    team = ratings["TM"].where(major).astype("object")
    park = team.map(lambda t: team_names.get(t, t if t in table.index else np.nan) if isinstance(t, str) else np.nan)
    return pd.DataFrame(
        {"ID": ids.to_numpy(), "park": park.to_numpy(), "team": team.to_numpy(), "weight": 1.0, "expected": major}
    )


def report_unmatched_parks(stints: pd.DataFrame, label: str, league: str, season: int) -> None:
    """Log the teams that should have a park but are not in the ballpark csv (treated as neutral)."""
    unmatched = stints[stints["expected"] & stints["park"].isna()]
    if unmatched.empty:
        return
    share = unmatched["weight"].sum() / stints.loc[stints["expected"], "weight"].sum() * 100
    logger.warning(
        "%s %s %s: no ballpark for %s (%.2f%% of the playing time); treated as neutral",
        league,
        season,
        label,
        sorted(map(str, unmatched["team"].unique())),
        share,
    )


def effective_factors(stints: pd.DataFrame, table: pd.DataFrame, columns: list[str], home_share: float) -> pd.DataFrame:
    """Playing-time weighted effective factor per player ID. Unknown parks are neutral."""
    weights = stints["weight"].astype(float)
    own = pd.DataFrame(index=stints.index)
    for column in columns:
        park_factor = stints["park"].map(table[column]).astype(float)
        own[column] = (home_share * park_factor + (1 - home_share)).fillna(1.0)
    totals = weights.groupby(stints["ID"]).sum()
    weighted = own.mul(weights, axis=0).groupby(stints["ID"]).sum()
    result = weighted.div(totals.where(totals > 0), axis=0)
    # a player with no playing time counts as neutral
    return result.fillna(1.0)


def build_park_factors(
    league: str,
    season: int,
    source: str,
    df_ratings: pd.DataFrame,
    df_batting: pd.DataFrame,
    df_pitching: pd.DataFrame,
) -> ParkFactors | None:
    """Effective park factors for every batter and pitcher, or None when the save has no ballpark file."""
    save = get_save(league)
    path = ballpark_file(save, season)
    if path is None:
        return None
    table = load_ballparks(path)

    if source == "dump":
        team_parks, team_label = dump_team_parks(league, season, table)
        batting_stints = dump_stints(league, season, "batting", team_parks, team_label)
        pitching_stints = dump_stints(league, season, "pitching", team_parks, team_label)
    else:
        batting_stints = export_stints(df_batting["ID"], df_ratings, table, save.team_names)
        pitching_stints = export_stints(df_pitching["ID"], df_ratings, table, save.team_names)

    report_unmatched_parks(batting_stints, "batting", league, season)
    report_unmatched_parks(pitching_stints, "pitching", league, season)
    return ParkFactors(
        batting=effective_factors(batting_stints, table, BATTING_FACTORS, save.home_share),
        pitching=effective_factors(pitching_stints, table, PITCHING_FACTORS, save.home_share),
    )
