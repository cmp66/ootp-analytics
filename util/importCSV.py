import pandas as pd
import os


def import_csv_flex(
    league: str, season: int, basedir
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:

    filedir = f"{basedir}/files/{league}/{season}"
    fielding_index = 0
    batting_index = 0
    pitching_index = 0
    fielding_dfs = [None] * 27
    batting_dfs = [None] * 10
    pitching_dfs = [None] * 10

    for file in [
        f for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))
    ]:
        df_temp = pd.read_csv(f"{filedir}/{file}")

        # check if the PI/PA column exists
        if "CERA" in df_temp.columns:
            df_team_fielding = df_temp
        elif "PPG" in df_temp.columns:
            df_team_pitching = df_temp
        elif "wRC" in df_temp.columns:
            df_team_batting = df_temp
        elif "PI/PA" in df_temp.columns:
            batting_dfs[batting_index] = (
                batting_dfs[batting_index - 1].merge(df_temp, how="outer")
                if batting_index != 0
                else df_temp
            )
            batting_index += 1
        elif "IRS" in df_temp.columns:
            pitching_dfs[pitching_index] = (
                pitching_dfs[pitching_index - 1].merge(df_temp, how="outer")
                if pitching_index != 0
                else df_temp
            )
            pitching_index += 1
        elif "BIZ-R" in df_temp.columns:
            fielding_dfs[fielding_index] = (
                fielding_dfs[fielding_index - 1].merge(df_temp, how="outer")
                if fielding_index != 0
                else df_temp
            )
            fielding_index += 1

    df_player_fielding_stats = fielding_dfs[fielding_index - 1]
    df_player_batting_stats = batting_dfs[batting_index - 1]
    df_player_pitching_stats = pitching_dfs[pitching_index - 1]

    # Need to rename columns which have the same name in the two table (batting and ratings)
    df_player_batting_stats.rename(
        columns={"1B": "SINGLE", "2B": "DOUBLE", "3B": "TRIPLE", "BABIP": "BABIP-O"},
        inplace=True,
    )
    df_player_pitching_stats.rename(
        columns={"1B": "SINGLE", "2B": "DOUBLE", "3B": "TRIPLE"}, inplace=True
    )

    return (
        df_player_batting_stats,
        df_player_pitching_stats,
        df_player_fielding_stats,
        df_team_batting,
        df_team_pitching,
        df_team_fielding,
    )


def import_ratings(
    league: str, season: int, ratings_type: str, basedir: str
) -> pd.DataFrame:

    df_player_ratings = None

    filedir = f"{basedir}/files/{league}/{season}/ratings"
    print(os.listdir(filedir))
    print(f"Reading ratings from {filedir} of type {ratings_type}")
    for file in [
        f
        for f in os.listdir(filedir)
        if ratings_type in f and os.path.isfile(os.path.join(filedir, f))
    ]:
        print(f"Reading {file}")
        df_temp = pd.read_csv(f"{filedir}/{file}")
        df_player_ratings = (
            df_temp
            if df_player_ratings is None
            else df_player_ratings.merge(df_temp, how="outer")
        )

    df_player_ratings.rename(columns={"R": "ROOKIE"}, inplace=True)
    df_player_ratings = df_player_ratings.drop(columns=["POS", "EXP"])

    return df_player_ratings
