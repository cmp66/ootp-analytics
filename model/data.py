"""Training tables for the value models, built from the processed output files of a save.

One function per value stream (plans/14-models.md). Every table has the same shape: one row per player and season
(split tables: per player, season and pitcher/batter side), the ratings that are the features, a target, a weight (the
volume the target is based on), and the player id / season for the splits. Rows with a missing or infinite value are
dropped and counted in the log; nothing is replaced by 0."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model import domains
from util import paths

logger = logging.getLogger(__name__)

MIN_SIDE_PA = 30  # a vs-right / vs-left batting row needs this many PA against that side
MIN_SIDE_IP = 10  # and this many innings for a pitching split row
MIN_SIDE_BIP = 40


@dataclass
class TrainingTable:
    name: str
    frame: pd.DataFrame
    features: list[str]
    target: str = "target"
    weight: str = "weight"
    groups: str = "ID"
    season: str = "season"
    meta: dict = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target]

    @property
    def w(self) -> pd.Series:
        return self.frame[self.weight]

    def __len__(self) -> int:
        return len(self.frame)


def read_output(save: str, season: int, kind: str) -> pd.DataFrame:
    path = paths.output_file(save, season, kind)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/process_seasons.py {save} --season {season}")
    return pd.read_csv(path, low_memory=False)


def read_ratings(save: str, season: int, columns: list[str]) -> pd.DataFrame:
    """The player-data ratings as numbers, one row per ID; '-' (not applicable) becomes NaN."""
    ratings = read_output(save, season, "player-data")
    missing = [c for c in columns if c not in ratings.columns]
    if missing:
        raise KeyError(f"{save} {season}: ratings columns not in player-data: {missing}")
    wanted = ["ID", *[c for c in dict.fromkeys(columns) if c != "ID"]]
    out = ratings[wanted].apply(pd.to_numeric, errors="coerce")
    return out.drop_duplicates("ID").set_index("ID")


def _seasons(seasons) -> list[int]:
    return [int(s) for s in seasons]


def _clean(name: str, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Drop rows with a missing or infinite feature, target or weight and log how many."""
    check = frame[[*features, "target", "weight"]].replace([np.inf, -np.inf], np.nan)
    bad = check.isna().any(axis=1)
    if bad.any():
        logger.warning("%s: dropped %d of %d rows with a missing or infinite value", name, bad.sum(), len(frame))
    return frame[~bad & (frame["weight"] > 0)].reset_index(drop=True)


def _stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _table(name, frames, features, meta=None) -> TrainingTable:
    frame = _clean(name, _stack(frames), features)
    return TrainingTable(name=name, frame=frame, features=list(features), meta=meta or {})


# ---------------------------------------------------------------- batting and baserunning


def batting_table(save: str, seasons, features: list[str], min_pa: int = domains.MIN_PA) -> TrainingTable:
    """Target: park adjusted batting runs (wRAAAdj) per 600 PA; weight: PA."""
    frames = []
    for season in _seasons(seasons):
        hitting = read_output(save, season, "hitting")
        hitting = hitting[hitting["PA"] >= min_pa]
        frame = pd.DataFrame(
            {
                "ID": hitting["ID"],
                "season": season,
                "target": hitting["wRAAAdj"] / hitting["PA"] * 600,
                "weight": hitting["PA"].astype(float),
            }
        )
        frames.append(frame.join(read_ratings(save, season, features), on="ID"))
    return _table(f"{save}-{domains.BATTING}", frames, features)


def baserunning_table(save: str, seasons, features: list[str], min_pa: int = domains.MIN_PA) -> TrainingTable:
    """Target: baserunning runs (BSR) per 600 PA; weight: PA."""
    frames = []
    for season in _seasons(seasons):
        hitting = read_output(save, season, "hitting")
        hitting = hitting[hitting["PA"] >= min_pa]
        frame = pd.DataFrame(
            {
                "ID": hitting["ID"],
                "season": season,
                "target": hitting["BSR"] / hitting["PA"] * 600,
                "weight": hitting["PA"].astype(float),
            }
        )
        frames.append(frame.join(read_ratings(save, season, features), on="ID"))
    return _table(f"{save}-{domains.BASERUNNING}", frames, features)


def _side_columns(ratings: pd.DataFrame, pairs: dict, side: str) -> pd.DataFrame:
    """The rating of each pair for one side, under the pair's common name."""
    index = 0 if side == domains.RIGHT else 1
    return pd.DataFrame({name: ratings[columns[index]] for name, columns in pairs.items()}, index=ratings.index)


def batting_side_features(ratings: pd.DataFrame, side: str, shared=()) -> pd.DataFrame:
    """Features of the stacked batting model for one pitcher side, from ratings indexed by player ID.

    The training tables and the scoring code both call this, so the two cannot drift apart."""
    out = _side_columns(ratings, domains.BATTING_SIDE_PAIRS, side)
    out["side_left"] = float(side == domains.LEFT)
    out["bats_left"] = (ratings["B"] == 2).astype(float)
    out["bats_switch"] = (ratings["B"] == 3).astype(float)
    for extra in shared:
        out[extra] = ratings[extra]
    return out


def batting_split_rating_columns(shared=()) -> list[str]:
    return [c for pair in domains.BATTING_SIDE_PAIRS.values() for c in pair] + ["B", *shared]


def pitching_side_features(ratings: pd.DataFrame, side: str, shared=()) -> pd.DataFrame:
    """Features of the stacked pitching model for one batter side, from ratings indexed by player ID."""
    out = _side_columns(ratings, domains.PITCHING_SIDE_PAIRS, side)
    out["side_left"] = float(side == domains.LEFT)
    out["throws_left"] = (ratings["T"] == 2).astype(float)
    for extra in shared:
        out[extra] = ratings[extra]
    return out


def pitching_split_rating_columns(shared=()) -> list[str]:
    return [c for pair in domains.PITCHING_SIDE_PAIRS.values() for c in pair] + ["T", *shared]


BATTING_SPLIT_FEATURES = [*domains.BATTING_SIDE_PAIRS, "side_left", "bats_left", "bats_switch"]
PITCHING_SPLIT_FEATURES = [*domains.PITCHING_SIDE_PAIRS, "side_left", "throws_left"]


def batting_split_table(save: str, seasons, shared: list[str] = (), min_pa: int = MIN_SIDE_PA) -> TrainingTable:
    """Stacked platoon rows: one per player, season and pitcher side (vs right / vs left).

    Features: the batter's ratings against that side (BA, GAP, POW, EYE, K), side_left, bats_left, bats_switch and any
    `shared` ratings. Target: wRAA against that side per 600 PA; weight: PA against that side."""
    frames = []
    for season in _seasons(seasons):
        ratings = read_ratings(save, season, batting_split_rating_columns(shared))
        for side, kind, column in [
            (domains.RIGHT, "right-hitting", "wRAARight"),
            (domains.LEFT, "left-hitting", "wRAALeft"),
        ]:
            split = read_output(save, season, kind)
            split = split[split["PA"] >= min_pa]
            frame = pd.DataFrame(
                {
                    "ID": split["ID"],
                    "season": season,
                    "target": split[column] / split["PA"] * 600,
                    "weight": split["PA"].astype(float),
                }
            )
            frames.append(frame.join(batting_side_features(ratings, side, shared), on="ID"))
    return _table(f"{save}-{domains.BATTING}-split", frames, [*BATTING_SPLIT_FEATURES, *shared])


# ---------------------------------------------------------------- pitching


def pitching_fip_table(
    save: str, seasons, features: list[str], min_ip: float = domains.MIN_IP, roles: tuple[str, ...] | None = None
) -> TrainingTable:
    """Target: park adjusted FIP runs (FIPRunsAdj) per 200 IP; weight: IP.

    roles keeps only pitchers whose usage bucket (PBaseline) is in the tuple, for example ("SP", "SP-lean")."""
    frames = []
    for season in _seasons(seasons):
        pitching = read_output(save, season, "pitching")
        pitching = pitching[pitching["IPClean"] >= min_ip]
        if roles is not None:
            pitching = pitching[pitching["PBaseline"].isin(roles)]
        frame = pd.DataFrame(
            {
                "ID": pitching["ID"],
                "season": season,
                "target": pitching["FIPRunsAdj"] / pitching["IPClean"] * 200,
                "weight": pitching["IPClean"].astype(float),
                "start_share": (pitching["GS"] / pitching["G"]).where(pitching["G"] > 0),
            }
        )
        frames.append(frame.join(read_ratings(save, season, features), on="ID"))
    return _table(f"{save}-{domains.PITCHING_FIP}", frames, features)


def pitching_babip_table(save: str, seasons, features: list[str], min_bip: int = domains.MIN_BIP) -> TrainingTable:
    """Target: BABIP on balls in play minus the season's league BABIP; weight: balls in play (BIP)."""
    frames = []
    for season in _seasons(seasons):
        pitching = read_output(save, season, "pitching")
        league = pitching["HitsBIP"].sum() / pitching["BIP"].sum()
        pitching = pitching[pitching["BIP"] >= min_bip]
        frame = pd.DataFrame(
            {
                "ID": pitching["ID"],
                "season": season,
                "target": pitching["HitsBIP"] / pitching["BIP"] - league,
                "weight": pitching["BIP"].astype(float),
            }
        )
        frames.append(frame.join(read_ratings(save, season, features), on="ID"))
    return _table(f"{save}-{domains.PITCHING_BABIP}", frames, features)


def pitching_split_table(save: str, seasons, part: str = "fip", shared: list[str] = ()) -> TrainingTable:
    """Stacked platoon rows for pitchers: one per pitcher, season and batter side (vs right / vs left batters).

    part "fip": target FIPRuns200 against that side, weight IP; part "babip": BABIP minus the side's league BABIP,
    weight BIP. Features: the pitcher's ratings against that side (STU, CON, PBABIP, HRR), side_left, throws_left and
    any `shared` ratings."""
    if part not in ("fip", "babip"):
        raise ValueError("part must be 'fip' or 'babip'")
    frames = []
    for season in _seasons(seasons):
        ratings = read_ratings(save, season, pitching_split_rating_columns(shared))
        for side, kind in [(domains.RIGHT, "right-pitching"), (domains.LEFT, "left-pitching")]:
            split = read_output(save, season, kind)
            if part == "fip":
                split = split[split["IPClean"] >= MIN_SIDE_IP]
                target, weight = split["FIPRuns200"], split["IPClean"].astype(float)
            else:
                league = split["HitsBIP"].sum() / split["BIP"].sum()
                split = split[split["BIP"] >= MIN_SIDE_BIP]
                target, weight = split["HitsBIP"] / split["BIP"] - league, split["BIP"].astype(float)
            frame = pd.DataFrame({"ID": split["ID"], "season": season, "target": target, "weight": weight})
            frames.append(frame.join(pitching_side_features(ratings, side, shared), on="ID"))
    return _table(f"{save}-pitching-{part}-split", frames, [*PITCHING_SPLIT_FEATURES, *shared])


# ---------------------------------------------------------------- fielding


def fielding_table(
    save: str, seasons, position: int, features: list[str], min_ip: float = domains.MIN_FIELDING_IP
) -> TrainingTable:
    """Target: fielding runs per full season (FldRunsSeason) at one position; weight: innings there."""
    if position not in domains.FIELD_POSITIONS:
        raise ValueError(f"position must be one of {domains.FIELD_POSITIONS}")
    frames = []
    for season in _seasons(seasons):
        fielding = read_output(save, season, "fielding")
        fielding = fielding[(fielding["POS"] == position) & (fielding["IPClean"] >= min_ip)]
        frame = pd.DataFrame(
            {
                "ID": fielding["ID"],
                "season": season,
                "target": fielding["FldRunsSeason"],
                "weight": fielding["IPClean"].astype(float),
            }
        )
        frames.append(frame.join(read_ratings(save, season, features), on="ID"))
    return _table(f"{save}-{domains.FIELDING}-{position}", frames, features, meta={"position": position})
