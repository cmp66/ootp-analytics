from dataclasses import dataclass

import pandas as pd

import stats
from stats import batting, fielding, parks, pitching
from util import importCSV
from util.config import get_save


@dataclass
class SeasonResult:
    ratings: pd.DataFrame
    batting: pd.DataFrame
    batting_right: pd.DataFrame
    batting_left: pd.DataFrame
    pitching: pd.DataFrame
    pitching_right: pd.DataFrame
    pitching_left: pd.DataFrame
    fielding: pd.DataFrame
    league_batting: pd.DataFrame
    player_data: pd.DataFrame


def process_season(league: str, season: int, source: str | None = None, ratings: str | None = None) -> SeasonResult:
    """Import one season and calculate and write every output file (each is written once)."""
    source = source or get_save(league).source

    (
        df_ratings,
        df_batting,
        df_batting_right,
        df_batting_left,
        df_pitching,
        df_pitching_right,
        df_pitching_left,
        df_fielding,
        *_,
    ) = importCSV.import_files(league, season, source, ratings, write_player_data=False)

    park_factors = parks.build_park_factors(league, season, source, df_ratings, df_batting, df_pitching)

    df_batting, df_lg_batting, df_batting_right, df_batting_left = batting.process_batting_data(
        df_batting,
        df_batting_right,
        df_batting_left,
        df_ratings,
        league,
        season,
        source,
        park_factors.batting if park_factors else None,
    )
    df_fielding = fielding.process_fielding_data(df_fielding, df_lg_batting, df_ratings, league, season)
    df_pitching = pitching.process_pitching_data(
        df_pitching, df_ratings, league, season, df_lg_batting, park_factors.pitching if park_factors else None
    )
    df_pitching_right, df_pitching_left = pitching.process_pitching_splits(
        df_pitching_right, df_pitching_left, df_pitching, league, season, df_lg_batting
    )
    df_player_data = stats.add_league_attribute_averages(league, season, df_ratings)

    return SeasonResult(
        ratings=df_ratings,
        batting=df_batting,
        batting_right=df_batting_right,
        batting_left=df_batting_left,
        pitching=df_pitching,
        pitching_right=df_pitching_right,
        pitching_left=df_pitching_left,
        fielding=df_fielding,
        league_batting=df_lg_batting,
        player_data=df_player_data,
    )
