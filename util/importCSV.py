import pandas as pd


def import_csv(
    league: str, season: int, base_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    batting_file = f"{base_path}/files/{league}/{season}/{league}-{season}-Hitting.csv"
    pitching_file = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Pitching.csv"
    )
    fielding_file_C = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-C.csv"
    )
    fielding_file_1B = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-1B.csv"
    )
    fielding_file_2B = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-2B.csv"
    )
    fielding_file_3B = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-3B.csv"
    )
    fielding_file_SS = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-SS.csv"
    )
    fielding_file_LF = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-LF.csv"
    )
    fielding_file_CF = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-CF.csv"
    )
    fielding_file_RF = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-RF.csv"
    )
    fielding_file_P = (
        f"{base_path}/files/{league}/{season}/{league}-{season}-Fielding-P.csv"
    )

    df_player_battings_stats = pd.read_csv(batting_file)
    df_player_pitching_stats = pd.read_csv(pitching_file)

    df_player_fielding_stats_C = pd.read_csv(fielding_file_C)
    df_player_fielding_stats_1B = df_player_fielding_stats_C.merge(
        pd.read_csv(fielding_file_1B), how="outer"
    )
    df_player_fielding_stats_2B = df_player_fielding_stats_1B.merge(
        pd.read_csv(fielding_file_2B), how="outer"
    )
    df_player_fielding_stats_3B = df_player_fielding_stats_2B.merge(
        pd.read_csv(fielding_file_3B), how="outer"
    )
    df_player_fielding_stats_SS = df_player_fielding_stats_3B.merge(
        pd.read_csv(fielding_file_SS), how="outer"
    )
    df_player_fielding_stats_LF = df_player_fielding_stats_SS.merge(
        pd.read_csv(fielding_file_LF), how="outer"
    )
    df_player_fielding_stats_CF = df_player_fielding_stats_LF.merge(
        pd.read_csv(fielding_file_CF), how="outer"
    )
    df_player_fielding_stats_RF = df_player_fielding_stats_CF.merge(
        pd.read_csv(fielding_file_RF), how="outer"
    )
    df_player_fielding_stats = df_player_fielding_stats_RF.merge(
        pd.read_csv(fielding_file_P), how="outer"
    )

    return df_player_battings_stats, df_player_pitching_stats, df_player_fielding_stats
