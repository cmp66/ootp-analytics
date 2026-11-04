"""The data the screens share, loaded once for the whole viewer.

A cache per screen would hold the same league several times, and `st.cache_data` hands every rerun a fresh copy. These
use `st.cache_resource` (one object for all pages and sessions, never copied) and keep at most `MAX_LEAGUES` leagues at a time, so
the memory of the viewer stays at one or two leagues however many screens there are. The objects are shared: a screen must not
change them (`league.park_frame` and the other functions that adjust values work on a copy).
"""

import numpy as np
import pandas as pd
import streamlit as st

from util import paths
from viewer import queries

MAX_LEAGUES = 2


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


def player_row(save: str, season: int, player_id: int) -> pd.Series:
    """The ratings row of one player, named by his ID."""
    table = player_data(save, season)
    row = table[table["ID"] == player_id].iloc[0]
    return row.rename(player_id)
