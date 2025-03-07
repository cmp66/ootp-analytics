from pandas import DataFrame
import math

positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
of_positions = [7, 8, 9]
if_positions = [3, 4, 5, 6]
dp_positions = [4, 6]


def get_season_adjustment(pos: int) -> int:
    return 220 if pos == 1 else 900 if pos == 2 else 1200


def get_fielding_adjustment(pos: str) -> int:
    return 220 if pos == 1 else 1000 if pos == 2 else 1200


def get_dp_adjustment(pos: int) -> int:
    return 0.6 if pos in [4] else 0.4 if pos in [6] else 0.0


def get_out_run_value(pos: str) -> float:
    return 0.9 if pos in [7, 8, 9] else 0.75


def get_out_run_value_2B(pos: str) -> float:
    return 0.75 if pos in [4] else 0.0


def convert_fielding_position(pos: int) -> str:
    if pos in [1]:
        return "P"
    elif pos in [2]:
        return "C"
    elif pos in [3]:
        return "1B"
    elif pos in [4]:
        return "2B"
    elif pos in [5]:
        return "3B"
    elif pos in [6]:
        return "SS"
    elif pos in [7]:
        return "LF"
    elif pos in [8]:
        return "CF"
    elif pos in [9]:
        return "RF"
    else:
        return "DH"


def calculate_league_totals(
    df_player_stats: DataFrame,
) -> tuple[dict[str, float], DataFrame]:

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
    df_positional_totals = DataFrame({"POS": positions})
    df_positional_totals.set_index("POS", inplace=True)

    for position in positions:
        df_positional_totals.loc[position, "PlayPercent"] = (
            position_group.get_group(position)["PlaysMade"].sum()
            / position_group.get_group(position)["PlaysAttempted"].sum()
            * 100
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
            * 100
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
    return df_lg_fielding_stat, df_positional_totals.round(2)


def calculate_league_attribute_averages(df_player_stats: DataFrame) -> DataFrame:

    df_fielding_averages = DataFrame()

    for position in of_positions:
        filter = df_player_stats["POS"] == position
        iptotal = df_player_stats.loc[filter, "IPClean"].sum()

        df_player_stats.loc[filter, "Weight"] = (
            df_player_stats.loc[filter, "IPClean"] / iptotal
        )
        df_player_stats.loc[filter, "RangeWeight"] = (
            df_player_stats.loc[filter, "Weight"]
            * df_player_stats.loc[filter, "OF RNG"]
        )
        df_player_stats.loc[filter, "ErrorWeight"] = (
            df_player_stats.loc[filter, "Weight"]
            * df_player_stats.loc[filter, "OF ERR"]
        )
        df_player_stats.loc[filter, "ARMWeight"] = (
            df_player_stats.loc[filter, "Weight"]
            * df_player_stats.loc[filter, "OF ARM"]
        )

        df_fielding_averages.loc[position, "Range"] = df_player_stats.loc[
            filter, "RangeWeight"
        ].sum()
        df_fielding_averages.loc[position, "Error"] = df_player_stats.loc[
            filter, "ErrorWeight"
        ].sum()
        df_fielding_averages.loc[position, "Arm"] = df_player_stats.loc[
            filter, "ARMWeight"
        ].sum()
        df_fielding_averages.loc[position, "DP"] = 0

    for position in if_positions + [1]:
        filter = df_player_stats["POS"] == position
        iptotal = df_player_stats.loc[filter, "IPClean"].sum()

        df_player_stats.loc[filter, "Weight"] = (
            df_player_stats.loc[filter, "IPClean"] / iptotal
        )
        df_player_stats.loc[filter, "RangeWeight"] = (
            df_player_stats.loc[filter, "Weight"]
            * df_player_stats.loc[filter, "IF RNG"]
        )
        df_player_stats.loc[filter, "ErrorWeight"] = (
            df_player_stats.loc[filter, "Weight"]
            * df_player_stats.loc[filter, "IF ERR"]
        )
        df_player_stats.loc[filter, "ARMWeight"] = (
            df_player_stats.loc[filter, "Weight"]
            * df_player_stats.loc[filter, "IF ARM"]
        )
        df_player_stats.loc[filter, "DPWeight"] = (
            df_player_stats.loc[filter, "Weight"] * df_player_stats.loc[filter, "TDP"]
        )

        df_fielding_averages.loc[position, "Range"] = df_player_stats.loc[
            filter, "RangeWeight"
        ].sum()
        df_fielding_averages.loc[position, "Error"] = df_player_stats.loc[
            filter, "ErrorWeight"
        ].sum()
        df_fielding_averages.loc[position, "Arm"] = df_player_stats.loc[
            filter, "ARMWeight"
        ].sum()
        df_fielding_averages.loc[position, "DP"] = df_player_stats.loc[
            filter, "DPWeight"
        ].sum()

    filter = df_player_stats["POS"] == 2
    iptotal = df_player_stats.loc[filter, "IPClean"].sum()
    df_player_stats.loc[filter, "Weight"] = (
        df_player_stats.loc[filter, "IPClean"] / iptotal
    )
    df_player_stats.loc[filter, "FRMWeight"] = (
        df_player_stats.loc[filter, "Weight"] * df_player_stats.loc[filter, "C FRM"]
    )
    df_player_stats.loc[filter, "BLKWeight"] = (
        df_player_stats.loc[filter, "Weight"] * df_player_stats.loc[filter, "C ABI"]
    )
    df_player_stats.loc[filter, "ARMWeight"] = (
        df_player_stats.loc[filter, "Weight"] * df_player_stats.loc[filter, "C ARM"]
    )
    df_fielding_averages.loc[2, "Frame"] = df_player_stats.loc[
        filter, "FRMWeight"
    ].sum()
    df_fielding_averages.loc[2, "Block"] = df_player_stats.loc[
        filter, "BLKWeight"
    ].sum()
    df_fielding_averages.loc[2, "Arm"] = df_player_stats.loc[filter, "ARMWeight"].sum()

    return df_fielding_averages.round(2)


def calculate_player_fielding_stats(
    df_player_stats: DataFrame, df_player_ratings: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:

    # df_player_stats["POS"] = df_player_stats["POS"].apply(convert_fielding_position)

    df_player_stats["DP"] = df_player_stats["DP"] * df_player_stats["POS"].apply(
        get_dp_adjustment
    )

    df_player_stats["ARM"] = df_player_stats["ARM"] * df_player_stats["POS"].apply(
        lambda x: 1 if x in of_positions else 0
    )

    df_player_stats = df_player_stats.merge(df_player_ratings, on="ID")

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

    df_player_stats["runsPAdjSeason"] = (
        df_player_stats["wPMAAAdj"]
        / df_player_stats["IPClean"]
        * df_player_stats["POS"].apply(get_fielding_adjustment)
    )

    df_fielding_attribute_averages = calculate_league_attribute_averages(
        df_player_stats
    )

    df_player_stats["IFRngDelta"] = df_player_stats["IF RNG"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Range"])
    df_player_stats["IFArmDelta"] = df_player_stats["IF ARM"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Arm"])
    df_player_stats["IFErrDelta"] = df_player_stats["IF ERR"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Error"])
    df_player_stats["IFTDPDelta"] = df_player_stats["TDP"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["DP"])
    df_player_stats["OFRngDelta"] = df_player_stats["OF RNG"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Range"])
    df_player_stats["OFArmDelta"] = df_player_stats["OF ARM"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Arm"])
    df_player_stats["OFERRDelta"] = df_player_stats["OF ERR"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Error"])
    df_player_stats["CFRMDelta"] = df_player_stats["C FRM"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Frame"])
    df_player_stats["CBLKDelta"] = df_player_stats["C ABI"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Block"])
    df_player_stats["CARMDelta"] = df_player_stats["C ARM"] - df_player_stats[
        "POS"
    ].apply(lambda x: df_fielding_attribute_averages.loc[x]["Arm"])

    for x in [
        "IFRngDelta",
        "IFArmDelta",
        "IFErrDelta",
        "IFTDPDelta",
        "OFRngDelta",
        "OFArmDelta",
        "OFERRDelta",
        "CFRMDelta",
        "CBLKDelta",
        "CARMDelta",
    ]:
        df_player_stats = df_player_stats.astype({x: "float"})
        df_player_stats = df_player_stats.round({x: 2})

    df_player_stats = df_player_stats.dropna(subset=["runsPAdj", "runsPAdjSeason"])

    columns_to_remove = [
        x for x in df_player_ratings.columns if x not in ["ID", "POS", "Name"]
    ]
    df_player_stats.drop(columns_to_remove, axis=1, inplace=True)
    df_player_stats.set_index("ID", inplace=True)

    return df_positional_totals, df_fielding_attribute_averages, df_player_stats
