"""The data the screens share, loaded once for the whole viewer.

A cache per screen would hold the same league several times, and `st.cache_data` hands every rerun a fresh copy. These
use `st.cache_resource` (one object for all pages and sessions, never copied) and keep at most `MAX_LEAGUES` leagues at a time, so
the memory of the viewer stays at one or two leagues however many screens there are. The objects are shared: a screen must not
change them (`league.park_frame` and the other functions that adjust values work on a copy).
"""

import ctypes
import sys

import numpy as np
import pandas as pd
import streamlit as st

from util import paths
from viewer import queries

MAX_LEAGUES = 2
M_MMAP_THRESHOLD = -3  # mallopt parameter of glibc
MMAP_THRESHOLD = 128 * 1024


def fix_mmap_threshold() -> bool:
    """Give big arrays back to the operating system when they are freed (Linux, glibc only; True when it was set).

    By default glibc raises the size above which memory is taken straight from the system (up to 32 MB) each time such a block is
    freed, so after a few pages the tables that pandas makes and drops stay in the process: the viewer peaked at 525 MB, with
    only about 70 MB of tables alive. Fixing the threshold turns that off (peak 414 MB with every page opened)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return bool(ctypes.CDLL("libc.so.6").mallopt(M_MMAP_THRESHOLD, MMAP_THRESHOLD))
    except OSError, AttributeError:
        return False


fix_mmap_threshold()


def shrink(frame: pd.DataFrame) -> pd.DataFrame:
    """The same table with 32-bit floats: half the memory, far more precision than the values (runs, ratings) need."""
    floats = frame.select_dtypes("float64").columns
    return frame.astype(
        {column: np.float32 for column in floats}
    ).copy()  # the copy joins the columns into a few blocks


@st.cache_resource(max_entries=MAX_LEAGUES, show_spinner="Loading the player files...")
def outlook(save: str, season: int) -> pd.DataFrame:
    """One row per player and ranked slot (queries.load_outlook)."""
    return shrink(queries.load_outlook(save, season))


@st.cache_resource(max_entries=MAX_LEAGUES, show_spinner=False)
def finances(save: str, season: int):
    return queries.load_finances(save, season)


@st.cache_resource(max_entries=MAX_LEAGUES, show_spinner=False)
def player_data(save: str, season: int) -> pd.DataFrame:
    """The processed ratings of every player (one row each)."""
    table = pd.read_csv(paths.output_file(save, season, "player-data"), low_memory=False)
    return shrink(table.drop_duplicates("ID"))


@st.cache_resource(max_entries=MAX_LEAGUES, show_spinner=False)
def growth(save: str, season: int) -> pd.DataFrame | None:
    """ID, d_cur, d_pot and days: recent growth from the archive of exports (model/growth.py); None until the archive has two
    exports 90 days apart."""
    path = paths.output_file(save, season, "growth")
    return shrink(pd.read_csv(path)) if path.exists() else None


def player_row(save: str, season: int, player_id: int) -> pd.Series:
    """The ratings row of one player, named by his ID."""
    table = player_data(save, season)
    row = table[table["ID"] == player_id].iloc[0]
    return row.rename(player_id)
