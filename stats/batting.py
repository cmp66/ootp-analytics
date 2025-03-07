from pandas import DataFrame
from stats import leagueAdjustments


def calculate_player_batting_stats(
    df_player_stats: DataFrame,
    df_lg_stat: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
) -> DataFrame:

    df_player_stats = df_player_stats.merge(df_player_ratings, on="ID")

    park_adjustments = leagueAdjustments.get_park_adjustments(league)

    df_player_stats["UBRAA"] = (
        df_player_stats["UBR"]
        - (
            (df_player_stats["SINGLE"] + df_player_stats["BB"] + df_player_stats["HP"])
            * 3
            + df_player_stats["DOUBLE"] * 2
            + df_player_stats["TRIPLE"]
            - df_player_stats["SB"]
            - df_player_stats["CS"] * 3
        )
        * df_lg_stat.loc["UBR"]["Value"]
    )

    df_player_stats["wOBA"] = (
        df_player_stats["SINGLE"] * df_lg_stat.loc["coef_1B"]["Value"]
        + df_player_stats["DOUBLE"] * df_lg_stat.loc["coef_2B"]["Value"]
        + df_player_stats["TRIPLE"] * df_lg_stat.loc["coef_3B"]["Value"]
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
                df_player_stats["SINGLE"]
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
    df_player_stats["wRAA600"] = df_player_stats["wRAA"] * 600 / df_player_stats["PA"]
    df_player_stats["OBP"] = (
        df_player_stats["H"] + df_player_stats["BB"] + df_player_stats["HP"]
    ) / (df_player_stats["PA"])
    df_player_stats["BSR"] = df_player_stats["UBRAA"] + df_player_stats["wSB"]
    df_player_stats["BSR600"] = df_player_stats["BSR"] * 600 / df_player_stats["PA"]
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
        df_player_stats["SINGLE"] + df_player_stats["BB"] + df_player_stats["HP"]
    )

    columns_to_remove = [
        x for x in df_player_ratings.columns if x not in ["ID", "POS", "Name"]
    ]
    df_player_stats.drop(columns_to_remove, axis=1, inplace=True)
    df_player_stats.set_index("ID", inplace=True)

    return df_player_stats
