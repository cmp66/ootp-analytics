"""Fine ages: how old a player is in years and days, from his date of birth and the in-game date of the ratings.

The exports and the dumps give the age as a whole number of years, and the export files do not say which in-game date they were
taken on. The date can be recovered from the players themselves: the age is the whole years between the date of birth and that
date, so every player says that the date is on or after his birthday of that age and before the next one. Over thousands of
players the window closes to one day (`export_date`). The `Sct` column of an export (the date a player was last scouted) is the
date of the ratings: for most leagues it is the export date, in some it is older (a player is only re-scouted now and then), and
a rating of 60 days ago belongs to the age of 60 days ago (`rating_dates`).
"""

import re

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def _birthday(dob: pd.Series, years: pd.Series) -> pd.Series:
    """The dates `years` after each date of birth (29 February counts as 28 February in years that have none)."""
    day = np.where((dob.dt.month == 2) & (dob.dt.day == 29), 28, dob.dt.day)
    return pd.to_datetime({"year": dob.dt.year + years.astype(int), "month": dob.dt.month, "day": day})


def date_window(dob: pd.Series, age: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The first and the last day on which every (date of birth, whole age) pair is true; None without usable pairs. The window
    is one day for a league of any size, and it is empty (first after last) when the ages do not fit one date."""
    dob = pd.to_datetime(dob, errors="coerce")
    age = pd.to_numeric(age, errors="coerce")
    ok = dob.notna() & age.notna()
    if not ok.any():
        return None
    dob, age = dob[ok].reset_index(drop=True), age[ok].reset_index(drop=True)
    first = _birthday(dob, age).max()
    last = _birthday(dob, age + 1).min() - pd.Timedelta(days=1)
    return first, last


def export_date(dob: pd.Series, age: pd.Series) -> pd.Timestamp | None:
    """The in-game date the ages were counted on: the first day of the window of `date_window` (None when there is none)."""
    window = date_window(dob, age)
    return None if window is None else window[0]


def parse_game_date(text) -> pd.Timestamp:
    """A date as the game writes it ('Feb. 26th   2046'); NaT when it is not one."""
    if not isinstance(text, str):
        return pd.NaT
    cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", text).replace(".", "")
    return pd.to_datetime(re.sub(r"\s+", " ", cleaned).strip(), format="%b %d %Y", errors="coerce")


def rating_dates(scouted: pd.Series, export: pd.Timestamp) -> pd.Series:
    """The date of each player's ratings: his `Sct` date, the export date where it is missing and never after the export."""
    parsed = scouted.map(parse_game_date)
    return parsed.fillna(export).clip(upper=export)


def fine_age(dob: pd.Series, on) -> pd.Series:
    """Age in years with the fraction on a date (or on each player's own date when `on` is a Series): the whole years since the
    date of birth plus the share of the current year of life that has passed, so the whole part is the age the game shows.
    """
    dob = pd.to_datetime(dob, errors="coerce")
    on = pd.Series(pd.Timestamp(on), index=dob.index) if not isinstance(on, pd.Series) else pd.to_datetime(on)
    years = on.dt.year - dob.dt.year
    years = years - (_birthday(dob, years) > on).astype(int)  # the birthday of this year is still to come
    last, following = _birthday(dob, years), _birthday(dob, years + 1)
    return years + (on - last).dt.days / (following - last).dt.days
