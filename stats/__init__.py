import pandas as pd
from util import paths


def add_league_attribute_averages(
    league: str, season: int, df_player_data: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add the league average rating columns to the player data and write player-data.csv.

    df_player_data is the cleaned ratings frame (indexed by ID). When omitted it is read from the
    player-data csv written by importCSV.import_files."""

    if df_player_data is None:
        df_player_data = pd.read_csv(paths.output_file(league, season, "player-data"))
    else:
        df_player_data = df_player_data.reset_index()

    df_woba = pd.read_csv(paths.output_file(league, season, "woba-calcs"))
    df_woba = df_woba.set_index("Stat")

    df_fielding = pd.read_csv(paths.output_file(league, season, "fielding-attributes"))
    df_fielding = df_fielding.set_index("POS")

    woba = df_woba["Value"]
    league_columns = {
        "lgBabipRating": woba["lgBabipRating"],
        "lgAvkRating": woba["lgAvkRating"],
        "lgConRating": woba["lgConRating"],
        "lgEyeRating": woba["lgEyeRating"],
        "lgPowRating": woba["lgPowRating"],
        "lgPbabipRating": woba["lgPbabipRating"],
        "lgStuRating": woba["lgStuRating"],
        "lgHrrRating": woba["lgHrrRating"],
        "lgCtlRating": woba["lgCtlRating"],
        "lgMovRating": woba["lgMovRating"],
    }
    # (column name, position, fielding-attributes column)
    for name, position, column in [
        ("lgCRating", 2, "PRat"),
        ("lg1BRating", 3, "PRat"),
        ("lg2BRating", 4, "PRat"),
        ("lg3BRating", 5, "PRat"),
        ("lgSSRating", 6, "PRat"),
        ("lgLFRating", 7, "PRat"),
        ("lgCFRating", 8, "PRat"),
        ("lgRFRating", 9, "PRat"),
        ("lg1BRange", 3, "Range"),
        ("lg2BRange", 4, "Range"),
        ("lg3BRange", 5, "Range"),
        ("lgSSRange", 6, "Range"),
        ("lgLFRange", 7, "Range"),
        ("lgCFRange", 8, "Range"),
        ("lgRFRange", 9, "Range"),
        ("lgRFError", 9, "Error"),
        ("lgCFError", 8, "Error"),
        ("lgLFError", 7, "Error"),
        ("lgSSError", 6, "Error"),
        ("lg3BError", 5, "Error"),
        ("lg2BError", 4, "Error"),
        ("lg1BError", 3, "Error"),
        ("lgRFArm", 9, "Arm"),
        ("lgCFArm", 8, "Arm"),
        ("lgLFArm", 7, "Arm"),
        ("lgCArm", 2, "Arm"),
        ("lgSSArm", 6, "Arm"),
        ("lg3BArm", 5, "Arm"),
        ("lg2BArm", 4, "Arm"),
        ("lg1BArm", 3, "Arm"),
        ("lg1BDP", 3, "DP"),
        ("lg2BDP", 4, "DP"),
        ("lg3BDP", 5, "DP"),
        ("lgSSDP", 6, "DP"),
        ("lgCFrm", 2, "Frame"),
        ("lgCBlock", 2, "Block"),
    ]:
        league_columns[name] = df_fielding.loc[position, column]

    df_player_data = df_player_data.assign(**league_columns)
    df_player_data.to_csv(paths.output_file(league, season, "player-data"), index=False)
    return df_player_data
