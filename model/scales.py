"""Which scale each rating column of a league is on: the usual 20-80 (steps of 5) or 1-10 (some leagues show ratings that way).

Only the values tell: a 1-10 column never goes above 10, a 20-80 column never below 20 (a missing rating, such as a pitch the
pitcher does not throw, is 0 on both). The number of pitches (`PIT`) and the height are not ratings and have no scale.

A 1-10 rating is a coarser bucketing of the same underlying rating as the 20-80 one, and where the bucket edges fall is different
for every rating column. Two dumps of one league taken at the same moment, one on each scale, show how often each 20-80 value is
each 1-10 value (`paired_counts`, kept in config/rating-scales/<version>.csv). With that table a 20-80 rating can be turned into a
1-10 one the way the game does it (`Conversion.to_1_10`, drawing the 1-10 value from the observed shares) and a 1-10 rating into
the 20-80 value expected for it (`Conversion.expected_20_80`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model import domains
from util import importCSV

S20_80, S1_10, OTHER = "20-80", "1-10", "other"
NOT_RATINGS = {"PIT", "HT"}  # the number of pitches and the first baseman height (cm)
MAX_1_10 = 10.0
MIN_20_80, MAX_20_80 = 20.0, 80.0


def rating_columns() -> list[str]:
    """The ratings the value models read (domains.REVIEWED_FEATURES), without the non-ratings."""
    columns = []
    for features in domains.REVIEWED_FEATURES.values():
        for group in features.values() if isinstance(features, dict) else [features]:
            columns += [c for c in group if c not in NOT_RATINGS and c not in columns]
    return columns


def column_scale(values: pd.Series) -> str:
    """ "1-10", "20-80" or "other" (empty, or values that fit neither) for one column; zeros and missing values are ignored."""
    seen = pd.to_numeric(values, errors="coerce")
    seen = seen[seen > 0]
    if seen.empty:
        return OTHER
    if seen.max() <= MAX_1_10:
        return S1_10
    if seen.min() >= MIN_20_80 and seen.max() <= MAX_20_80:
        return S20_80
    return OTHER


def profile(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One row per column present in `frame`: the scale, the range, how many different values, whether they are whole numbers
    and the step between the values (the smallest gap; 5 on the 20-80 scale)."""
    rows = []
    for column in columns or rating_columns():
        if column not in frame.columns:
            continue
        seen = pd.to_numeric(frame[column], errors="coerce").dropna()
        seen = seen[seen > 0]
        distinct = sorted(seen.unique())
        gaps = [b - a for a, b in zip(distinct, distinct[1:])]
        rows.append(
            {
                "column": column,
                "scale": column_scale(frame[column]),
                "min": seen.min() if len(seen) else None,
                "max": seen.max() if len(seen) else None,
                "distinct": len(distinct),
                "whole": bool((seen == seen.round()).all()) if len(seen) else None,
                "step": min(gaps) if gaps else None,
            }
        )
    return pd.DataFrame(rows)


def detect_scales(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, str]:
    """Column -> its scale, for the model ratings (or `columns`) found in `frame`."""
    return {column: column_scale(frame[column]) for column in (columns or rating_columns()) if column in frame.columns}


# ---------------------------------------------------------------- conversion between the scales

PAIR_FILES = {
    "players_batting.csv": importCSV.DUMP_BATTING_RENAME,
    "players_pitching.csv": importCSV.DUMP_PITCHING_RENAME,
    "players_fielding.csv": importCSV.DUMP_FIELDING_RENAME,
}
TABLE_DIR = Path(__file__).resolve().parent.parent / "config" / "rating-scales"
NOT_PAIRED = {"ID", "Team Name"}


def version_of(ratings: str) -> str:
    """The game version of a ratings set: Standard27 -> OOTP27."""
    digits = "".join(c for c in ratings if c.isdigit())
    if not digits:
        raise ValueError(f"cannot tell the game version of ratings set {ratings!r}")
    return f"OOTP{digits}"


def table_name(version: str, relative: bool = True) -> str:
    """The file stem of a conversion table: OOTP27 (relative ratings on) or OOTP27-direct (off). With relative ratings off the
    1-10 buckets are a fixed mapping of the underlying skill, with them on the game fits the buckets to the league."""
    return version if relative else f"{version}-direct"


def paired_counts(folder_20_80: Path, folder_1_10: Path, partial: bool = False) -> pd.DataFrame:
    """Rows column (the model's name), v20, v10, count: how many players have each pair of values in two dumps of one league taken
    at the same moment. Columns that are the same in both dumps (not ratings) are left out. The dumps must hold the same players
    (in any order); with `partial` the players in both are paired and the others left out (two runs of one league that were not
    exported from exactly the same state, e.g. Development-Analysis)."""
    rows = []
    for file, rename in PAIR_FILES.items():
        first, second = pd.read_csv(folder_20_80 / file), pd.read_csv(folder_1_10 / file)
        if not partial and set(first["player_id"]) != set(second["player_id"]):
            raise ValueError(f"{file}: the two dumps do not hold the same players")
        both = first.merge(second, on="player_id", suffixes=("_a", "_b"))
        for source, name in rename.items():
            if source not in first.columns or name in NOT_PAIRED or not pd.api.types.is_numeric_dtype(first[source]):
                continue
            a, b = both[source + "_a"], both[source + "_b"]
            if a.equals(b) or a.max() <= MAX_1_10:
                continue
            counts = pd.crosstab(a, b).stack()
            rows += [(name, int(a), int(b), int(n)) for (a, b), n in counts[counts > 0].items()]
    return pd.DataFrame(rows, columns=["column", "v20", "v10", "count"])


class Conversion:
    """The conversion between the scales of one game version, from the counts of paired_counts (or its csv)."""

    def __init__(self, counts: pd.DataFrame):
        self.counts = counts
        self.columns = sorted(counts["column"].unique())
        self._shares = {}
        self._expected = {}
        for column, group in counts.groupby("column"):
            wide = group.pivot_table(index="v20", columns="v10", values="count", aggfunc="sum", fill_value=0)
            self._shares[column] = wide.div(wide.sum(axis=1), axis=0)
            by_10 = group.assign(weighted=group["v20"] * group["count"]).groupby("v10")[["weighted", "count"]].sum()
            self._expected[column] = (by_10["weighted"] / by_10["count"]).to_dict()

    @classmethod
    def load(cls, version: str, folder: Path = TABLE_DIR, relative: bool = True) -> "Conversion":
        """The conversion table of a game version; `relative` False: the one for a league with relative ratings off."""
        path = folder / f"{table_name(version, relative)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/build_scale_tables.py")
        return cls(pd.read_csv(path))

    def to_1_10(self, values: pd.Series, column: str, rng: np.random.Generator) -> pd.Series:
        """20-80 ratings as 1-10: each value becomes one of the 1-10 values it has in the paired dumps, with their shares.
        Zero (no rating) and missing values stay as they are; a value the table never saw takes the nearest one."""
        shares = self._shares[column]
        known = shares.index.to_numpy()
        out = values.astype(float).copy()
        draw = rng.random(len(values))
        for value in pd.unique(values.dropna()):
            if value == 0:
                continue
            row = shares.loc[known[np.abs(known - value).argmin()]]
            mask = (values == value).to_numpy()
            cumulative = row.cumsum().to_numpy()
            pick = np.minimum((draw[mask][:, None] > cumulative[None, :]).sum(axis=1), len(row) - 1)
            out[mask] = row.index.to_numpy()[pick]
        return out

    def expected_20_80(self, values: pd.Series, column: str) -> pd.Series:
        """1-10 ratings as the average 20-80 value that rating has in the paired dumps (zero and missing stay as they are)."""
        table = self._expected[column]
        known = np.array(sorted(table))
        out = values.astype(float).copy()
        for value in pd.unique(values.dropna()):
            if value == 0:
                continue
            out[values == value] = table[known[np.abs(known - value).argmin()]]
        return out

    def coarsen_frame(self, frame: pd.DataFrame, columns, seed: int = 0) -> pd.DataFrame:
        """`frame` with the 20-80 columns among `columns` turned into 1-10 (see to_1_10); deterministic for a seed."""
        out = frame.copy()
        for i, column in enumerate(c for c in columns if c in frame.columns and c in self._shares):
            out[column] = self.to_1_10(frame[column], column, np.random.default_rng([seed, i]))
        return out

    def expect_frame(self, frame: pd.DataFrame, columns) -> pd.DataFrame:
        """`frame` with the 1-10 columns among `columns` turned into their expected 20-80 value."""
        out = frame.copy()
        for column in (c for c in columns if c in frame.columns and c in self._expected):
            out[column] = self.expected_20_80(frame[column], column)
        return out


# ---------------------------------------------------------------- profiles: models for a mix of scales


def model_set(ratings: str, profile: str | None = None) -> str:
    """The folder key of a model set: the ratings set, plus `@profile` for the models trained on a mix of scales."""
    return f"{ratings}@{profile}" if profile else ratings


def profile_columns(save: str, season: int | None = None, conversion: "Conversion | None" = None) -> list[str]:
    """The rating columns a league shows on the 1-10 scale (and the conversion table knows), from its latest player-data."""
    from util import paths
    from util.seasons import available_seasons

    latest = season or available_seasons(save)[-1]
    frame = pd.read_csv(paths.output_file(save, latest, "player-data"), low_memory=False)
    known = set(conversion.columns) if conversion is not None else None
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in NOT_RATINGS]
    return [c for c in numeric if (known is None or c in known) and column_scale(frame[c]) == S1_10]


def coarse_view(conversion: "Conversion", columns, back: bool = False, seed: int = 0):
    """A ratings view for model.data.ratings_view: the 20-80 ratings of the training league as 1-10 (and, with back, as the 20-80
    value expected for that), for the columns of a profile; the draw is fixed by the seed and the season."""

    def transform(ratings: pd.DataFrame, season: int) -> pd.DataFrame:
        coarse = conversion.coarsen_frame(ratings, columns, seed=seed * 10000 + season)
        return conversion.expect_frame(coarse, columns) if back else coarse

    return transform
