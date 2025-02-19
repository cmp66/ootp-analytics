from pandas import DataFrame
from stats import leagueAdjustments


def calculate_player_batting_stats(
    df_player_stats: DataFrame, df_lg_stat: DataFrame, league: str
) -> DataFrame:

    park_adjustments = leagueAdjustments.get_park_adjustments(league)

    df_player_stats["UBRAA"] = (
        df_player_stats["UBR"]
        - (
            (df_player_stats["1B"] + df_player_stats["BB"] + df_player_stats["HP"]) * 3
            + df_player_stats["2B"] * 2
            + df_player_stats["3B"]
            - df_player_stats["SB"]
            - df_player_stats["CS"] * 3
        )
        * df_lg_stat.loc["UBR"]["Value"]
    )

    df_player_stats["wOBA"] = (
        df_player_stats["1B"] * df_lg_stat.loc["coef_1B"]["Value"]
        + df_player_stats["2B"] * df_lg_stat.loc["coef_2B"]["Value"]
        + df_player_stats["3B"] * df_lg_stat.loc["coef_3B"]["Value"]
        + df_player_stats["HR"] * df_lg_stat.loc["coef_HR"]["Value"]
        + (df_player_stats["BB"] - df_player_stats["IBB"])
        * df_lg_stat.loc["coef_BB"]["Value"]
        + df_player_stats["HP"] * df_lg_stat.loc["coef_HP"]["Value"]
    ) / df_player_stats["PA"]

    df_player_stats["wSB"] = (
        (df_player_stats["SB"] * df_lg_stat.loc["run_value_sb"]["Value"])
        + (df_player_stats["CS"] * df_lg_stat.loc["run_value_cs"]["Value"])
        - (
            df_lg_stat.loc["wSB"]["Value"]
            * (
                df_player_stats["1B"]
                + df_player_stats["BB"]
                + df_player_stats["HP"]
                - df_player_stats["IBB"]
            )
        )
    )

    df_player_stats["wRAA"] = (
        (df_player_stats["wOBA"] - df_lg_stat.loc["wOBA"]["Value"])
        / df_lg_stat.loc["wOBA_SCALE"]["Value"]
    ) * df_player_stats["PA"]
    df_player_stats["OBP"] = (
        df_player_stats["H"] + df_player_stats["BB"] + df_player_stats["HP"]
    ) / (df_player_stats["PA"])
    df_player_stats["BSR"] = df_player_stats["UBRAA"] + df_player_stats["wSB"]
    df_player_stats["OFF"] = df_player_stats["BSR"] + df_player_stats["wRAA"]
    df_player_stats["OFFAdj"] = (df_player_stats["OFF"]) + (
        df_player_stats["ORG"].apply(
            lambda x: park_adjustments[x] if x in park_adjustments else 0
        )
        / 600
        * df_player_stats["PA"]
    )
    df_player_stats["OFF600"] = df_player_stats["OFF"] * 600 / df_player_stats["PA"]
    df_player_stats["SBAPERCENT"] = (df_player_stats["SB"] + df_player_stats["CS"]) / (
        df_player_stats["1B"] + df_player_stats["BB"] + df_player_stats["HP"]
    )

    return df_player_stats
