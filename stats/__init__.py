import pandas as pd
from util import paths


def add_league_attribute_averages(league: str, season: int):

    df_player_data = pd.read_csv(paths.output_file(league, season, "player-data"))

    df_woba = pd.read_csv(paths.output_file(league, season, "woba-calcs"))
    df_woba = df_woba.set_index("Stat")

    df_fielding = pd.read_csv(paths.output_file(league, season, "fielding-attributes"))
    df_fielding = df_fielding.set_index("POS")

    df_player_data["lgBabipRating"] = df_woba.loc["lgBabipRating"]["Value"]
    df_player_data["lgAvkRating"] = df_woba.loc["lgAvkRating"]["Value"]
    df_player_data["lgConRating"] = df_woba.loc["lgConRating"]["Value"]
    df_player_data["lgEyeRating"] = df_woba.loc["lgEyeRating"]["Value"]
    df_player_data["lgPowRating"] = df_woba.loc["lgPowRating"]["Value"]
    df_player_data["lgPbabipRating"] = df_woba.loc["lgPbabipRating"]["Value"]
    df_player_data["lgStuRating"] = df_woba.loc["lgStuRating"]["Value"]
    df_player_data["lgHrrRating"] = df_woba.loc["lgHrrRating"]["Value"]
    df_player_data["lgCtlRating"] = df_woba.loc["lgCtlRating"]["Value"]
    df_player_data["lgMovRating"] = df_woba.loc["lgMovRating"]["Value"]
    df_player_data["lgCRating"] = df_fielding.loc[2]["PRat"]
    df_player_data["lg1BRating"] = df_fielding.loc[3]["PRat"]
    df_player_data["lg2BRating"] = df_fielding.loc[4]["PRat"]
    df_player_data["lg3BRating"] = df_fielding.loc[5]["PRat"]
    df_player_data["lgSSRating"] = df_fielding.loc[6]["PRat"]
    df_player_data["lgLFRating"] = df_fielding.loc[7]["PRat"]
    df_player_data["lgCFRating"] = df_fielding.loc[8]["PRat"]
    df_player_data["lgRFRating"] = df_fielding.loc[9]["PRat"]

    df_player_data["lg1BRange"] = df_fielding.loc[3]["Range"]
    df_player_data["lg2BRange"] = df_fielding.loc[4]["Range"]
    df_player_data["lg3BRange"] = df_fielding.loc[5]["Range"]
    df_player_data["lgSSRange"] = df_fielding.loc[6]["Range"]
    df_player_data["lgLFRange"] = df_fielding.loc[7]["Range"]
    df_player_data["lgCFRange"] = df_fielding.loc[8]["Range"]
    df_player_data["lgRFRange"] = df_fielding.loc[9]["Range"]
    df_player_data["lgRFError"] = df_fielding.loc[9]["Error"]
    df_player_data["lgCFError"] = df_fielding.loc[8]["Error"]
    df_player_data["lgLFError"] = df_fielding.loc[7]["Error"]
    df_player_data["lgSSError"] = df_fielding.loc[6]["Error"]
    df_player_data["lg3BError"] = df_fielding.loc[5]["Error"]
    df_player_data["lg2BError"] = df_fielding.loc[4]["Error"]
    df_player_data["lg1BError"] = df_fielding.loc[3]["Error"]
    df_player_data["lgRFArm"] = df_fielding.loc[9]["Arm"]
    df_player_data["lgCFArm"] = df_fielding.loc[8]["Arm"]
    df_player_data["lgLFArm"] = df_fielding.loc[7]["Arm"]
    df_player_data["lgCArm"] = df_fielding.loc[2]["Arm"]
    df_player_data["lgSSArm"] = df_fielding.loc[6]["Arm"]
    df_player_data["lg3BArm"] = df_fielding.loc[5]["Arm"]
    df_player_data["lg2BArm"] = df_fielding.loc[4]["Arm"]
    df_player_data["lg1BArm"] = df_fielding.loc[3]["Arm"]
    df_player_data["lg1BDP"] = df_fielding.loc[3]["DP"]
    df_player_data["lg2BDP"] = df_fielding.loc[4]["DP"]
    df_player_data["lg3BDP"] = df_fielding.loc[5]["DP"]
    df_player_data["lgSSDP"] = df_fielding.loc[6]["DP"]
    df_player_data["lgCFrm"] = df_fielding.loc[2]["Frame"]
    df_player_data["lgCBlock"] = df_fielding.loc[2]["Block"]

    df_player_data.to_csv(
        paths.output_file(league, season, "player-data"),
        index=False,
    )
