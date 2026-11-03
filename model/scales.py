"""Which scale each rating column of a league is on: the usual 20-80 (steps of 5) or 1-10 (some leagues show ratings that way).

Only the values tell: a 1-10 column never goes above 10, a 20-80 column never below 20 (a missing rating, such as a pitch the
pitcher does not throw, is 0 on both). The number of pitches (`PIT`) and the height are not ratings and have no scale.
"""

import pandas as pd

from model import domains

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
