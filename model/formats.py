"""Conversions between what an OOTP dump holds and what an in-game export shows (plan 16, stage 1b).

The long training leagues come as database dumps with numbers; the human leagues only exist as in-game exports with
letters and categories. A model that should score both must use something both can give:
- personality traits (work ethic WE, intelligence INT, leadership LEA, loyalty LOY, finances FIN): 1-200 numbers in the
  dump, L / N / H in the export. Adaptability (AD) is only in the export and cannot be modelled.
- proneness: 1-200 numbers in the dump (higher = more injury prone, checked against missed playing time), five categories in
  the export (Iron Man < Durable < Normal < Fragile < Wrecked).
- level: a numeric team level in the dump, a text level in the export.
- how a player entered: draft fields and the college flag in the dump; draft year, round and pick in the export (no college
  flag: high school and college draftees are told apart by age at the draft).

The cut points that turn dump numbers into the export's categories are not known (the game does not publish them and no
save exists in both formats), so they are estimated by matching shares: the lowest share of the dump values that the
export labels L is L, and so on. `fit_bounds` / `categorize` do that; `config/outlook.toml` keeps the result per ratings set.
"""

import numpy as np
import pandas as pd

TRAIT_LABELS = ("L", "N", "H")
PRONE_LABELS = ("Iron Man", "Durable", "Normal", "Fragile", "Wrecked")  # increasing proneness
TRAITS = ("WE", "INT", "LEA", "LOY", "FIN")
PRONE = "Prone"
PRONE3_LABELS = (
    "Durable",
    "Normal",
    "Fragile",
)  # Iron Man and Wrecked fall at the 1 / 200 limits, which share-matching cannot split
PRONE3 = {"Iron Man": "Durable", "Durable": "Durable", "Normal": "Normal", "Fragile": "Fragile", "Wrecked": "Fragile"}
AGE_AT_DRAFT_COLLEGE = (
    20  # drafted at this age or older = college (checked on the dump, see scripts/check_format_parity.py)
)

_TRAIT_ALIASES = {"1": "L", "2": "N", "U": "N", "3": "H"}
_PRONE_ALIASES = {
    "IRON MAN": "Iron Man",
    "DURABLE": "Durable",
    "NORMAL": "Normal",
    "FRAGILE": "Fragile",
    "WRECKED": "Wrecked",
}

DUMP_LEVEL_GROUP = {1: "MLB", 2: "AAA", 3: "AA", 4: "A", 5: "A", 6: "Rookie"}
EXPORT_LEVEL_GROUP = {
    "FA": "FA",
    "MLB": "MLB",
    "AAA": "AAA",
    "AA": "AA",
    "A+": "A",
    "A-": "A",
    "A": "A",
    "R+": "Rookie",
    "R": "Rookie",
    "R-": "Rookie",
}


def trait_label(value) -> str | float:
    """L / N / H from an export value ('L', 'N', 'H', or the numbers 1-3 some exports use); NaN when unknown."""
    text = str(value).strip().upper()
    return text if text in TRAIT_LABELS else _TRAIT_ALIASES.get(text, np.nan)


def prone_label(value) -> str | float:
    return _PRONE_ALIASES.get(str(value).strip().upper(), np.nan)


def label_shares(categories: pd.Series, labels) -> dict[str, float]:
    """Share of each label among the known categories (zero for labels that do not occur)."""
    known = categories[categories.isin(labels)]
    if known.empty:
        raise ValueError(f"no category among {labels} to take shares from")
    return {label: float((known == label).mean()) for label in labels}


def fit_bounds(values: pd.Series, shares: dict[str, float], labels) -> list[float]:
    """Upper bound of every label but the last, so that the dump values fall into the labels in the shares given.

    `labels` run from the lowest values to the highest; a value equal to a bound belongs to the lower label. The dump
    numbers pile up at their limits (1 and 200) and at multiples of 5, so the bound is the value whose cumulative share
    is closest to the target (a bound below the minimum gives an empty first label)."""
    clean = np.sort(values.dropna().to_numpy(dtype=float))
    if len(clean) == 0:
        raise ValueError("no values to fit cut points on")
    distinct = np.unique(clean)
    candidates = np.concatenate([[distinct[0] - 1.0], distinct])
    cdf = np.concatenate([[0.0], np.searchsorted(clean, distinct, side="right") / len(clean)])
    cumulative = np.cumsum([shares[label] for label in labels])[:-1]
    return [float(candidates[int(np.argmin(np.abs(cdf - share)))]) for share in cumulative]


def categorize(values: pd.Series, bounds: list[float], labels) -> pd.Series:
    """Labels for numbers: <= bounds[0] is labels[0], and so on; NaN stays NaN."""
    index = np.searchsorted(np.asarray(bounds, dtype=float), values.to_numpy(dtype=float), side="left")
    out = pd.Series(np.asarray(labels, dtype=object)[index], index=values.index)
    return out.where(values.notna())


def ordinal(categories: pd.Series, labels) -> pd.Series:
    """0 for the first label, 1 for the next, ...; NaN for unknown categories."""
    return categories.map({label: i for i, label in enumerate(labels)}).astype(float)


def category_means(values: pd.Series, categories: pd.Series) -> dict[str, float]:
    """Mean dump value of each category: the number to use for a category when a numeric model scores an export."""
    frame = pd.DataFrame({"value": values, "category": categories}).dropna()
    return {str(label): float(group["value"].mean()) for label, group in frame.groupby("category")}


def level_group(level, source: str, free_agent: bool = False) -> str:
    """Coarse level (MLB, AAA, AA, A, Rookie, FA, other) from a dump level number or an export level text.

    A dump free agent still has the team he left, so he is told apart by the flag."""
    if free_agent:
        return "FA"
    if source == "dump":
        try:
            return DUMP_LEVEL_GROUP.get(int(level), "other")
        except TypeError, ValueError:
            return "other"
    return EXPORT_LEVEL_GROUP.get(str(level).strip(), "other")


def draft_route(age_at_draft) -> str:
    """High school or college draft from the age at the draft (the export has no college flag)."""
    return "college_draft" if age_at_draft >= AGE_AT_DRAFT_COLLEGE else "hs_draft"


def export_level_number(level, ratings: str) -> float:
    """The numeric team level of the dump (1 MLB, 2 AAA, 3 AA, 4 A, 6 Rookie; OOTP 26 also 5 for low A) from an export level.

    Free agents, winter league and international rows have no level (NaN), as in the dump."""
    text = str(level).strip()
    low_a = 5 if ratings == "Standard26" else 4  # OOTP 26 has a separate low-A level
    numbers = {"MLB": 1, "AAA": 2, "AA": 3, "A+": 4, "A": 4, "A-": low_a, "R+": 6, "R": 6, "R-": 6}
    return float(numbers.get(text, np.nan))
