"""The plan of an organization, shared by the pages that show it (Level moves and the depth chart): the moves of every player
(viewer/roster.py) and who plays where at the levels they end up at (viewer/lineup.py)."""

import pandas as pd
import streamlit as st

from model import scales  # noqa: F401  (the version of the ratings is passed in by the pages)
from util.config import get_save
from viewer import data_cache, injuries, lineup, minors, playing, roster


def load(save: str, season: int):
    return data_cache.outlook(save, season), data_cache.player_data(save, season), data_cache.growth(save, season)


@st.cache_data(show_spinner="Working out the moves...", max_entries=4, ttl=3600)
def planned(
    save: str,
    season: int,
    org: str,
    rules: minors.Rules,
    version: str,
    mlb_limit: int,
    settings: playing.Playing,
    injury: injuries.Injuries | None = None,
) -> tuple[roster.Plan, pd.DataFrame, pd.DataFrame]:
    """The plan of the organization, who plays where at the levels it ends up with, and the playing value of every slot each
    player could play there. The levels are chosen together with the slots when `settings.joint_weight` is above 0. A player
    on the injured list (`injury`, viewer/injuries.py) is left out: no roster spot, no move, no slot, until he is well.
    """
    frame, ratings, growth = load(save, season)
    levels = get_save(save).levels
    mlb = roster.mlb_table(frame, ratings, org, rules, version)
    peers = roster.mlb_peers(frame, ratings, rules)
    table = minors.manage(frame, ratings, org, levels, rules, version)
    gone = injuries.listed(injuries.marks(data_cache.injured(save, season), injury or injuries.load_injuries()))
    mlb, table = (t[~t["ID"].isin(gone)].reset_index(drop=True) if "ID" in t else t for t in (mlb, table))
    joint = lineup.joint(frame, ratings, rules, settings, growth=growth) if settings.joint_weight > 0 else None
    result = roster.plan(table, frame, levels, rules, mlb, mlb_limit, peers, playing=joint)
    placed, offered = lineup.placement(result.players, frame, ratings, levels, rules, settings, growth)
    if growth is not None and not placed.empty:
        placed = placed.merge(growth[["ID", "d_cur"]].rename(columns={"d_cur": "Grow"}), on="ID", how="left")
    return result, placed, offered
