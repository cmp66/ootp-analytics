"""Injuries: how long a player is out, and what that means in the viewer. Pure functions, no UI.

The export says `INJ` = Yes and `Left`, the time left as text ("4 days", "one week", "2-3 months", "?"). Three states:

* **DTD** (day to day): `day_to_day` days or fewer. He is marked, stays where he is and still plays.
* **OUT**: more than that, up to `injured_list` days. Marked, not moved and not cover for a position, but still in the charts.
* **IL** (injured list): more than `injured_list` days, or a time that is not known. In the sim he goes on the injured list: he is
  taken off the depth charts and out of the plan (no roster spot, no lineup slot) until an export says he is well again.

A range counts by its upper end ("2-3 weeks" is 21 days). Settings: `[injuries]` in config/minors.toml.
"""

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from viewer.minors import MINORS_CONFIG

DTD, OUT, IL = "DTD", "OUT", "IL"
LABELS = {DTD: "DTD", OUT: "Out", IL: "IL"}
COLORS = {DTD: "#d4a017", OUT: "#e0457b", IL: "#8f6bd6"}  # gold, rose, violet
DAYS = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
PATTERN = re.compile(r"(\d+|one|two|three|four|five|six)(?:\s*-\s*(\d+))?\s*(day|week|month|year)s?")


@dataclass(frozen=True)
class Injuries:
    day_to_day: float = 7.0
    injured_list: float = 14.0


def load_injuries(path: Path = MINORS_CONFIG) -> Injuries:
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    values = dict(raw.get("injuries", {}))
    unknown = set(values) - set(Injuries.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown settings in [injuries]: {sorted(unknown)}")
    return replace(Injuries(), **{k: float(v) for k, v in values.items()})


def days(left) -> float:
    """The days left of an injury text, by the upper end of a range; infinity for a time that is not known ("?")."""
    text = str(left).strip().lower()
    found = PATTERN.fullmatch(text)
    if not found:
        return np.inf
    first, last, unit = found.groups()
    number = float(last if last else WORDS.get(first, first))
    return number * DAYS[unit]


def status(left_days: float, settings: Injuries) -> str:
    return DTD if left_days <= settings.day_to_day else OUT if left_days <= settings.injured_list else IL


def raw(frame: pd.DataFrame) -> pd.DataFrame:
    """The injured players of a frame with the export's `INJ` and `Left` columns, indexed by ID: `left` (the text) and `days`.
    Empty without the columns (an export that has no injuries)."""
    empty = pd.DataFrame(
        {"left": pd.Series(dtype=object), "days": pd.Series(dtype=float)}, index=pd.Index([], name="ID")
    )
    if not {"INJ", "Left"} <= set(frame.columns):
        return empty
    hurt = frame.loc[frame["INJ"].astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"]), ["ID", "Left"]]
    hurt = hurt.drop_duplicates("ID").set_index("ID")
    if hurt.empty:
        return empty
    return pd.DataFrame({"left": hurt["Left"].astype(str), "days": hurt["Left"].map(days).astype(float)})


def marks(injured: pd.DataFrame, settings: Injuries) -> pd.DataFrame:
    """`raw` with the state (`status`: DTD, OUT or IL) and the text for a table (`text`: "DTD 4 days", "Out 2 weeks", "IL ?")."""
    out = injured.copy()
    out["status"] = [status(d, settings) for d in out["days"]]
    out["text"] = [f"{LABELS[s]} {left}" for s, left in zip(out["status"], out["left"])]
    return out


def listed(marks_: pd.DataFrame | None) -> set:
    """The IDs on the injured list: they are not in the charts or the plan."""
    return set() if marks_ is None or marks_.empty else set(marks_.index[marks_["status"] == IL])


def add(frame: pd.DataFrame, marks_: pd.DataFrame | None, column: str = "Inj") -> pd.DataFrame:
    """`frame` (with an ID column) with the injury text of every injured player in `column`, next to the player's name;
    unchanged when nobody in it is hurt."""
    if marks_ is None or marks_.empty or "ID" not in frame.columns or column in frame.columns:
        return frame
    text = frame["ID"].map(marks_["text"])
    if text.isna().all():
        return frame
    out = frame.copy()
    out.insert(
        out.columns.get_loc("Player") + 1 if "Player" in out.columns else len(out.columns), column, text.fillna("")
    )
    return out


def css(text) -> str:
    key = str(text).split(" ")[0].upper()
    return f"color: {COLORS[key]}; font-weight: 700" if key in COLORS else ""


def paint(style, frame: pd.DataFrame, columns: tuple[str, ...] = ("Player", "Inj")):
    """Color the player (and injury) cells of a styled table by the state in its `Inj` column; no change without the column."""
    if "Inj" not in frame.columns:
        return style
    colors = [css(t) for t in frame["Inj"]]
    for column in columns:
        if column in frame.columns:
            style = style.apply(lambda col: colors, subset=[column])
    return style


def legend_html() -> str:
    """One line for a page: what the colors of injured players mean."""
    names = {DTD: "day to day", OUT: "out", IL: "injured list (off the charts and the plan)"}
    parts = [f'<span style="color:{COLORS[s]};font-weight:700">{LABELS[s]}</span> {names[s]}' for s in COLORS]
    return '<div style="font-size:13px;opacity:.85">Injured: ' + ", ".join(parts) + "</div>"
