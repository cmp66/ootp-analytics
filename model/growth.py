"""Recent growth of the players of an export league, from the dated archive of its ratings exports (`files/<save>/archive/`).

The outlook of a prospect depends on how his ratings have been moving: with talent change randomness on, a player who gained
a rating point in a year gains about 0.8 more over the next three (hitters; 0.38 for pitchers), and the same holds for a rise of
his potential (docs/models.md). The exports have one season each, so the history is the archive: the earliest snapshot of the
season against the newest one.

The change is measured in standard deviations of the older snapshot, because the leagues show ratings on different scales (PBL27
is all 1-10), then made relative to the league (the mean change of all players of the kind is taken off: a relative league
rebalances its ratings at the start of the offseason and around opening day, which moves everybody) and turned into 20-80 points
(`POINTS_PER_SD`, about the spread of the ratings of professional players) per year. Under `MIN_DAYS` apart the two snapshots say
too little (ratings change in a few percent of the players in a week): no growth is given.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from util import paths

HITTER = ["CON", "GAP", "EYE", "K's", "POW", "BABIP"]
HITTER_POTENTIAL = ["CON P", "GAP P", "EYE P", "K P", "POW P"]
PITCHER = ["STU", "MOV", "HRA", "PBABIP", "CON_1"]
PITCHER_POTENTIAL = ["STU P", "MOV P", "HRA P", "PBABIP P", "CON_1 P"]
PITCHER_POSITIONS = {"SP", "RP", "CL", "P"}
MIN_DAYS = 90
POINTS_PER_SD = 10.0  # hand tuned: the standard deviation of the ratings of professional players on the 20-80 scale
CAP = 6.0  # points a year in either direction: a wider change is a re-scouting or an error, not growth
COLUMNS = ["ID", "POS", *HITTER, *HITTER_POTENTIAL, *PITCHER, *PITCHER_POTENTIAL]


def snapshots(save: str, season: int) -> list[tuple[pd.Timestamp, Path]]:
    """The archived exports of the season, oldest first: (in-game date, folder)."""
    base = paths.archive_dir(save, season, "")
    folders = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    out = []
    for folder in folders:
        date = pd.to_datetime(folder.name, errors="coerce")
        if pd.notna(date) and any(folder.glob("*.csv.gz")):
            out.append((date, folder))
    return out


def read_snapshot(folder: Path) -> pd.DataFrame:
    """The ratings of every player of an archived export (one row each) with the columns of growth, others left out."""
    frames = []
    for path in sorted(folder.glob("*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
            header = pd.read_csv(handle, nrows=0).columns
        if "ID" not in header:
            continue
        frames.append(pd.read_csv(path, usecols=[c for c in COLUMNS if c in header], low_memory=False))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("ID").set_index("ID")


def mean_standardized(frame: pd.DataFrame, columns: list[str], reference: pd.DataFrame) -> pd.Series:
    """The mean over the columns of each player's rating in standard deviations of the `reference` snapshot (a rating of 0 or a
    missing one is no rating)."""
    out = {}
    for column in columns:
        if column not in frame or column not in reference:
            continue
        ref = pd.to_numeric(reference[column], errors="coerce").where(lambda s: s > 0)
        values = pd.to_numeric(frame[column], errors="coerce").where(lambda s: s > 0)
        if ref.notna().sum() > 1 and ref.std() > 0:
            out[column] = (values - ref.mean()) / ref.std()
    return pd.DataFrame(out).mean(axis=1) if out else pd.Series(np.nan, index=frame.index)


def growth(older: pd.DataFrame, newer: pd.DataFrame, days: float) -> pd.DataFrame:
    """ID, d_cur, d_pot (20-80 points a year, relative to the league) and days, for the players in both snapshots."""
    ids = older.index.intersection(newer.index)
    old, new = older.loc[ids], newer.loc[ids]
    pitcher = old.get("POS", pd.Series("", index=ids)).isin(PITCHER_POSITIONS)
    out = pd.DataFrame(index=ids, columns=["d_cur", "d_pot"], dtype=float)
    for who, rated, potential in ((~pitcher, HITTER, HITTER_POTENTIAL), (pitcher, PITCHER, PITCHER_POTENTIAL)):
        if not who.any():
            continue
        for name, columns in (("d_cur", rated), ("d_pot", potential)):
            change = mean_standardized(new[who], columns, old[who]) - mean_standardized(old[who], columns, old[who])
            change = change - change.mean()  # the league's own shift (a rebalance) is not growth
            out.loc[who, name] = (change * POINTS_PER_SD * 365.0 / days).clip(-CAP, CAP)
    out["days"] = float(days)
    return out.rename_axis("ID").reset_index()


def archive_growth(save: str, season: int) -> pd.DataFrame | None:
    """Growth between the oldest and the newest archived export of the season; None without two exports MIN_DAYS or more apart."""
    found = snapshots(save, season)
    if len(found) < 2:
        return None
    (first, old_folder), (last, new_folder) = found[0], found[-1]
    days = (last - first).days
    if days < MIN_DAYS:
        return None
    return growth(read_snapshot(old_folder), read_snapshot(new_folder), days)
