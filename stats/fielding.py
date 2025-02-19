from pandas import DataFrame
import math

positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
of_positions = [7, 8, 9]
if_positions = [3, 4, 5, 6]
dp_positions = [4, 6]


def get_season_adjustment(pos: int) -> int:
    return 220 if pos == 1 else 900 if pos == 2 else 1200


def get_fielding_adjustment(pos: int) -> int:
    return 220 if pos == 1 else 1000 if pos == 2 else 1200


def get_out_run_value(pos: int) -> float:
    return 0.9 if pos in [7, 8, 9] else 0.75


def get_out_run_value_2B(pos: int) -> float:
    return 0.75 if pos in [4] else 0.0


def calculate_league_totals(df_player_stats: DataFrame) -> dict[str, float]:

    league_fielding_totals = {
        "IPClean": df_player_stats["IPClean"].sum(),
        "PlaysAttempted": df_player_stats["PlaysAttempted"].sum(),
        "PlaysMade": df_player_stats["PlaysMade"].sum(),
        "FRM": df_player_stats["FRM"].sum(),
        "ARM": df_player_stats["ARM"].sum(),
        "E": df_player_stats["E"].sum(),
        "DP": df_player_stats["DP"].sum(),
        "SBA": df_player_stats["SBA"].sum(),
        "RTO": df_player_stats["RTO"].sum(),
    }

    position_group = df_player_stats.groupby("POS")
    df_positional_totals = DataFrame({"POS": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    df_positional_totals.set_index("POS", inplace=True)

    for position in positions:
        df_positional_totals.loc[position, "PlayPercent"] = (
            position_group.get_group(position)["PlaysMade"].sum()
            / position_group.get_group(position)["PlaysAttempted"].sum()
        )
        df_positional_totals.loc[position, "PlayAttemptedPerSeason"] = (
            position_group.get_group(position)["PlaysAttempted"].sum()
            / position_group.get_group(position)["IPClean"].sum()
            * get_season_adjustment(position)
        )
        df_positional_totals.loc[position, "PlayPerSeason"] = (
            position_group.get_group(position)["PlaysMade"].sum()
            / position_group.get_group(position)["IPClean"].sum()
            * get_season_adjustment(position)
        )
        df_positional_totals.loc[position, "ErrorPerSeason"] = (
            position_group.get_group(position)["E"].sum()
            / position_group.get_group(position)["IPClean"].sum()
            * get_season_adjustment(position)
        )
        df_positional_totals.loc[position, "ErrorPercentage"] = (
            position_group.get_group(position)["E"].sum()
            / position_group.get_group(position)["PlaysMade"].sum()
        )

        if position in of_positions:
            df_positional_totals.loc[position, "ARM"] = (
                position_group.get_group(position)["ARM"].sum()
                / position_group.get_group(position)["PlaysAttempted"].sum()
                * df_positional_totals.loc[position, "PlayAttemptedPerSeason"]
            )
        else:
            df_positional_totals.loc[position, "ARM"] = 0

        if position in [1, 2]:
            df_positional_totals.loc[2, "SBA"] = (
                position_group.get_group(position)["SBA"].sum()
                / position_group.get_group(position)["IPClean"].sum()
                * 1000
            )
            df_positional_totals.loc[2, "RTO"] = (
                position_group.get_group(position)["RTO"].sum()
                / position_group.get_group(position)["SBA"].sum()
            )
            df_positional_totals.loc[2, "FRM"] = (
                position_group.get_group(position)["FRM"].sum()
                / position_group.get_group(position)["IPClean"].sum()
            )
        else:
            df_positional_totals.loc[position, "SBA"] = 0
            df_positional_totals.loc[position, "RTO"] = 0
            df_positional_totals.loc[position, "FRM"] = 0

        if position in if_positions:
            df_positional_totals.loc[position, "DP"] = (
                position_group.get_group(position)["DP"].sum()
                / position_group.get_group(position)["IPClean"].sum()
                * get_season_adjustment(position)
            )
        else:
            df_positional_totals.loc[position, "DP"] = 0

    df_lg_fielding_stat = DataFrame(
        list(league_fielding_totals.items()), columns=["Stat", "Value"]
    ).set_index("Stat")

    # print(df_positional_totals)
    return df_lg_fielding_stat, df_positional_totals


def calculate_player_fielding_stats(df_player_stats: DataFrame) -> DataFrame:

    df_player_stats["DP"] = df_player_stats["DP"] * df_player_stats["POS"].apply(
        lambda x: 1 if x in dp_positions else 0
    )
    df_player_stats["ARM"] = df_player_stats["ARM"] * df_player_stats["POS"].apply(
        lambda x: 1 if x in of_positions else 0
    )

    df_player_stats["PlaysAttempted"] = (
        df_player_stats["BIZ-R"]
        + df_player_stats["BIZ-L"]
        + df_player_stats["BIZ-E"]
        + df_player_stats["BIZ-U"]
        + df_player_stats["BIZ-Z"]
        + df_player_stats["BIZ-I"]
    )
    df_player_stats["PlaysMade"] = (
        df_player_stats["BIZ-Rm"]
        + df_player_stats["BIZ-Lm"]
        + df_player_stats["BIZ-Em"]
        + df_player_stats["BIZ-Um"]
        + df_player_stats["BIZ-Zm"]
    )
    df_player_stats["IPClean"] = df_player_stats["IP"].apply(
        lambda x: math.modf(x)[1] + math.modf(x)[0] * 3.33
    )

    df_player_stats["PlaysMadeSeason"] = (
        df_player_stats["PlaysMade"]
        / df_player_stats["IPClean"]
        * df_player_stats["POS"].apply(get_season_adjustment)
    )

    df_lg_batting_stat, df_positional_totals = calculate_league_totals(df_player_stats)
    df_player_stats["PlayMadeAASeason"] = df_player_stats[
        "PlaysMadeSeason"
    ] - df_player_stats["POS"].apply(
        lambda x: df_positional_totals.loc[x, "PlayPerSeason"]
    )
    df_player_stats["ARMAASeason"] = df_player_stats["ARM"] / df_player_stats[
        "IPClean"
    ] * df_player_stats["POS"].apply(get_season_adjustment) - df_player_stats[
        "POS"
    ].apply(
        lambda x: df_positional_totals.loc[x, "ARM"]
    )
    df_player_stats["FRMAASeason"] = (
        df_player_stats["FRM"]
        / df_player_stats["IPClean"]
        * df_player_stats["POS"].apply(get_season_adjustment)
        - df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "FRM"])
        * df_player_stats["IPClean"]
    )
    df_player_stats["EAASeason"] = df_player_stats["E"] / df_player_stats[
        "IPClean"
    ] * df_player_stats["POS"].apply(get_season_adjustment) - df_player_stats[
        "POS"
    ].apply(
        lambda x: df_positional_totals.loc[x, "ErrorPerSeason"]
    )
    df_player_stats["DPAASeason"] = df_player_stats["DP"] / df_player_stats[
        "IPClean"
    ] * df_player_stats["POS"].apply(get_season_adjustment) - df_player_stats[
        "POS"
    ].apply(
        lambda x: df_positional_totals.loc[x, "DP"]
    )

    df_player_stats["runsPSeason"] = (
        df_player_stats["PlayMadeAASeason"]
        * df_player_stats["POS"].apply(get_out_run_value)
        - df_player_stats["EAASeason"] * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["DPAASeason"]
        * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["FRMAASeason"]
        + df_player_stats["ARMAASeason"]
        * df_player_stats["POS"].apply(get_out_run_value)
    )

    df_player_stats["PlayMadeAANow"] = (
        df_player_stats["PlaysMade"]
        - df_player_stats["POS"].apply(
            lambda x: df_positional_totals.loc[x, "PlayPerSeason"]
        )
        / df_player_stats["POS"].apply(get_season_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["ARMAANow"] = (
        df_player_stats["ARM"]
        - df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "ARM"])
        / df_player_stats["POS"].apply(get_season_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["FRMAANow"] = (
        df_player_stats["FRM"]
        - df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "FRM"])
        * df_player_stats["IPClean"]
    )
    df_player_stats["EAANow"] = (
        df_player_stats["E"]
        - df_player_stats["POS"].apply(
            lambda x: df_positional_totals.loc[x, "ErrorPerSeason"]
        )
        / df_player_stats["POS"].apply(get_season_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["DPAANow"] = (
        df_player_stats["DP"]
        - df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "DP"])
        / df_player_stats["POS"].apply(get_season_adjustment)
        * df_player_stats["IPClean"]
    )

    df_player_stats["runsPNow"] = (
        df_player_stats["PlayMadeAANow"]
        * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["ARMAANow"] * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["FRMAANow"]
        + df_player_stats["DPAANow"] * df_player_stats["POS"].apply(get_out_run_value)
        - (df_player_stats["EAANow"] * df_player_stats["POS"].apply(get_out_run_value))
    )

    df_player_stats["PlaysASeason"] = (
        df_player_stats["PlaysAttempted"]
        / df_player_stats["IPClean"]
        * df_player_stats["POS"].apply(get_season_adjustment)
    )
    df_player_stats["PlayAAASeason"] = df_player_stats[
        "PlaysASeason"
    ] - df_player_stats["POS"].apply(
        lambda x: df_positional_totals.loc[x, "PlayAttemptedPerSeason"]
    )
    df_player_stats["PlayPercent"] = (
        df_player_stats["PlaysMade"] / df_player_stats["PlaysAttempted"]
    )
    df_player_stats["PlayPercentAA"] = df_player_stats["PlayPercent"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_positional_totals.loc[x, "PlayPercent"])

    df_player_stats["wPMSeason"] = (
        df_player_stats["POS"].apply(
            lambda x: df_positional_totals.loc[x, "PlayAttemptedPerSeason"]
        )
        * df_player_stats["PlayPercent"]
    )
    df_player_stats["wPMAASeason"] = df_player_stats["wPMSeason"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_positional_totals.loc[x, "PlayPerSeason"])
    df_player_stats["wRunsPSeason"] = (
        df_player_stats["wPMAASeason"] * df_player_stats["POS"].apply(get_out_run_value)
        - df_player_stats["EAASeason"] * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["ARMAASeason"]
        * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["FRMAASeason"]
        + df_player_stats["DPAASeason"]
        * df_player_stats["POS"].apply(get_out_run_value_2B)
    )
    df_player_stats["AdjustedPANow"] = (
        df_player_stats["POS"].apply(
            lambda x: df_positional_totals.loc[x, "PlayAttemptedPerSeason"]
        )
        / df_player_stats["POS"].apply(get_fielding_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["wPMAdj"] = (
        df_player_stats["AdjustedPANow"] * df_player_stats["PlayPercent"]
    )
    df_player_stats["wPMAAAdj"] = df_player_stats["wPMAdj"] - (
        df_player_stats["POS"].apply(
            lambda x: df_positional_totals.loc[x, "PlayPerSeason"]
        )
        / df_player_stats["POS"].apply(get_fielding_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["ARMAAAdj"] = df_player_stats["ARM"] - (
        df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "ARM"])
        / df_player_stats["POS"].apply(get_fielding_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["FRMAAAdj"] = df_player_stats["FRM"] - (
        df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "FRM"])
        * df_player_stats["IPClean"]
    )
    df_player_stats["EAAAdj"] = (
        df_player_stats["E"]
        - df_player_stats["POS"].apply(
            lambda x: df_positional_totals.loc[x, "ErrorPerSeason"]
        )
        / df_player_stats["POS"].apply(get_season_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["DPAAdj"] = (
        df_player_stats["DP"]
        - df_player_stats["POS"].apply(lambda x: df_positional_totals.loc[x, "DP"])
        / df_player_stats["POS"].apply(get_fielding_adjustment)
        * df_player_stats["IPClean"]
    )
    df_player_stats["runsPAdj"] = (
        df_player_stats["wPMAAAdj"] * df_player_stats["POS"].apply(get_out_run_value)
        - df_player_stats["EAAAdj"] * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["DPAAdj"] * 0.325
        + df_player_stats["ARMAAAdj"] * df_player_stats["POS"].apply(get_out_run_value)
        + df_player_stats["FRMAAAdj"]
    )

    return df_player_stats
