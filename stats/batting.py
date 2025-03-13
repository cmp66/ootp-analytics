from pandas import DataFrame
import pandas as pd
from stats import leagueAdjustments, woba


def calculate_player_batting_stats(
    df_player_stats_total: DataFrame,
    df_player_stats_right: DataFrame,
    df_player_stats_left: DataFrame,
    df_lg_stat: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
) -> tuple[DataFrame, DataFrame, DataFrame]:

    # test for dataframe with no data
    skip_handedness = df_player_stats_right.empty

    df_player_stats_total = df_player_stats_total.merge(df_player_ratings, on="ID")

    if not skip_handedness:
        df_player_stats_right = df_player_stats_right.merge(df_player_ratings, on="ID")
        df_player_stats_left = df_player_stats_left.merge(df_player_ratings, on="ID")

    park_adjustments = leagueAdjustments.get_park_adjustments(league)

    df_player_stats_total["UBRAA"] = (
        df_player_stats_total["UBR"]
        - (
            (
                df_player_stats_total["SINGLE"]
                + df_player_stats_total["BB"]
                + df_player_stats_total["HP"]
            )
            * 3
            + df_player_stats_total["DOUBLE"] * 2
            + df_player_stats_total["TRIPLE"]
            - df_player_stats_total["SB"]
            - df_player_stats_total["CS"] * 3
        )
        * df_lg_stat.loc["lgUBR"]["Value"]
    )

    df_player_stats_total["wOBA"] = (
        df_player_stats_total["SINGLE"] * df_lg_stat.loc["coef_1B"]["Value"]
        + df_player_stats_total["DOUBLE"] * df_lg_stat.loc["coef_2B"]["Value"]
        + df_player_stats_total["TRIPLE"] * df_lg_stat.loc["coef_3B"]["Value"]
        + df_player_stats_total["HR"] * df_lg_stat.loc["coef_HR"]["Value"]
        + (df_player_stats_total["BB"] - df_player_stats_total["IBB"])
        * df_lg_stat.loc["coef_BB"]["Value"]
        + df_player_stats_total["HP"] * df_lg_stat.loc["coef_HP"]["Value"]
    ) / df_player_stats_total["PA"]

    if not skip_handedness:
        df_player_stats_right["wOBARight"] = (
            df_player_stats_right["SINGLE"] * df_lg_stat.loc["coef_1B"]["Value"]
            + df_player_stats_right["DOUBLE"] * df_lg_stat.loc["coef_2B"]["Value"]
            + df_player_stats_right["TRIPLE"] * df_lg_stat.loc["coef_3B"]["Value"]
            + df_player_stats_right["HR"] * df_lg_stat.loc["coef_HR"]["Value"]
            + (df_player_stats_right["BB"] - df_player_stats_right["IBB"])
            * df_lg_stat.loc["coef_BB"]["Value"]
            + df_player_stats_right["HP"] * df_lg_stat.loc["coef_HP"]["Value"]
        ) / df_player_stats_right["PA"]

        df_player_stats_left["wOBALeft"] = (
            df_player_stats_left["SINGLE"] * df_lg_stat.loc["coef_1B"]["Value"]
            + df_player_stats_left["DOUBLE"] * df_lg_stat.loc["coef_2B"]["Value"]
            + df_player_stats_left["TRIPLE"] * df_lg_stat.loc["coef_3B"]["Value"]
            + df_player_stats_left["HR"] * df_lg_stat.loc["coef_HR"]["Value"]
            + (df_player_stats_left["BB"] - df_player_stats_left["IBB"])
            * df_lg_stat.loc["coef_BB"]["Value"]
            + df_player_stats_left["HP"] * df_lg_stat.loc["coef_HP"]["Value"]
        ) / df_player_stats_left["PA"]

    df_player_stats_total["wSB"] = (
        (df_player_stats_total["SB"] * df_lg_stat.loc["run_value_sb"]["Value"])
        + (df_player_stats_total["CS"] * df_lg_stat.loc["run_value_cs"]["Value"])
        - (
            df_lg_stat.loc["lgwSB"]["Value"]
            * (
                df_player_stats_total["SINGLE"]
                + df_player_stats_total["BB"]
                + df_player_stats_total["HP"]
                - df_player_stats_total["IBB"]
            )
        )
    )

    df_player_stats_total["wRAA"] = (
        (df_player_stats_total["wOBA"] - df_lg_stat.loc["lgwOBA"]["Value"])
        / df_lg_stat.loc["wOBA_SCALE"]["Value"]
    ) * df_player_stats_total["PA"]

    if not skip_handedness:
        df_player_stats_right["wRAARight"] = (
            (df_player_stats_right["wOBARight"] - df_lg_stat.loc["lgwOBA"]["Value"])
            / df_lg_stat.loc["wOBA_SCALE"]["Value"]
        ) * df_player_stats_right["PA"]
        df_player_stats_left["wRAALeft"] = (
            (df_player_stats_left["wOBALeft"] - df_lg_stat.loc["lgwOBA"]["Value"])
            / df_lg_stat.loc["wOBA_SCALE"]["Value"]
        ) * df_player_stats_left["PA"]
    df_player_stats_total["wRAA600"] = (
        df_player_stats_total["wRAA"] * 600 / df_player_stats_total["PA"]
    )

    if not skip_handedness:
        df_player_stats_right["wRAA600Right"] = (
            df_player_stats_right["wRAARight"] * 600 / df_player_stats_right["PA"]
        )
        df_player_stats_left["wRAA600Left"] = (
            df_player_stats_left["wRAALeft"] * 600 / df_player_stats_left["PA"]
        )
    df_player_stats_total["OBP"] = (
        df_player_stats_total["H"]
        + df_player_stats_total["BB"]
        + df_player_stats_total["HP"]
    ) / (df_player_stats_total["PA"])
    df_player_stats_total["BSR"] = (
        df_player_stats_total["UBRAA"] + df_player_stats_total["wSB"]
    )
    df_player_stats_total["BSR600"] = (
        df_player_stats_total["BSR"] * 600 / df_player_stats_total["PA"]
    )
    df_player_stats_total["OFF"] = (
        df_player_stats_total["BSR"] + df_player_stats_total["wRAA"]
    )

    if not skip_handedness:
        df_player_stats_right["OFFRight"] = (
            df_player_stats_total["BSR"] + df_player_stats_right["wRAARight"]
        )
        df_player_stats_left["OFFLeft"] = (
            df_player_stats_total["BSR"] + df_player_stats_left["wRAALeft"]
        )
    df_player_stats_total["OFFAdj"] = (df_player_stats_total["OFF"]) + (
        df_player_stats_total["ORG"].apply(
            lambda x: park_adjustments[x] if x in park_adjustments else 0
        )
        / 600
        * df_player_stats_total["PA"]
    )
    df_player_stats_total["OFF600"] = (
        df_player_stats_total["OFF"] * 600 / df_player_stats_total["PA"]
    )

    if not skip_handedness:
        df_player_stats_right["OFF600Right"] = (
            df_player_stats_right["OFFRight"] * 600 / df_player_stats_right["PA"]
        )
        df_player_stats_left["OFF600Left"] = (
            df_player_stats_left["OFFLeft"] * 600 / df_player_stats_left["PA"]
        )
    df_player_stats_total["SBAPERCENT"] = (
        df_player_stats_total["SB"] + df_player_stats_total["CS"]
    ) / (
        df_player_stats_total["SINGLE"]
        + df_player_stats_total["BB"]
        + df_player_stats_total["HP"]
    )

    columns_to_remove = [
        x for x in df_player_ratings.columns if x not in (["ID", "POS", "Name"])
    ]
    df_player_stats_total.drop(columns_to_remove, axis=1, inplace=True)

    if not skip_handedness:
        df_player_stats_right.drop(columns_to_remove, axis=1, inplace=True)
        df_player_stats_left.drop(columns_to_remove, axis=1, inplace=True)

    for stat in df_lg_stat.index.tolist():
        df_player_stats_total[stat] = df_lg_stat.loc[stat]["Value"]

        if not skip_handedness:
            df_player_stats_right[stat] = df_lg_stat.loc[stat]["Value"]
            df_player_stats_left[stat] = df_lg_stat.loc[stat]["Value"]

    df_player_stats_total.set_index("ID", inplace=True)

    if not skip_handedness:
        df_player_stats_right.set_index("ID", inplace=True)
        df_player_stats_left.set_index("ID", inplace=True)

    return df_player_stats_total, df_player_stats_right, df_player_stats_left


def process_batting_data(
    df_player_batting_stats: DataFrame,
    df_player_batting_right_stats: DataFrame,
    df_player_batting_left_stats: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
    season: int,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    league_batting_totals = {
        "PA": (
            df_player_batting_stats["PA"].sum()
            if source == "dump"
            else df_player_batting_stats["PA"].sum()
            + df_player_batting_stats["SH"].sum()
            + df_player_batting_stats["CI"].sum()
        ),
        "AB": df_player_batting_stats["AB"].sum(),
        "R": df_player_batting_stats["R"].sum(),
        "H": df_player_batting_stats["H"].sum(),
        "1B": df_player_batting_stats["SINGLE"].sum(),
        "2B": df_player_batting_stats["DOUBLE"].sum(),
        "3B": df_player_batting_stats["TRIPLE"].sum(),
        "HR": df_player_batting_stats["HR"].sum(),
        "BB": df_player_batting_stats["BB"].sum(),
        "IBB": df_player_batting_stats["IBB"].sum(),
        "HP": df_player_batting_stats["HP"].sum(),
        "SO": df_player_batting_stats["SO"].sum(),
        "SF": df_player_batting_stats["SF"].sum(),
        "SH": df_player_batting_stats["SH"].sum(),
        "GIDP": df_player_batting_stats["GIDP"].sum(),
        "SB": df_player_batting_stats["SB"].sum(),
        "CS": df_player_batting_stats["CS"].sum(),
        "UBR": df_player_batting_stats["UBR"].sum(),
    }

    df_lg_batting_stat = pd.DataFrame(
        list(woba.calc_league_data(league_batting_totals).items()),
        columns=["Stat", "Value"],
    )
    df_lg_batting_stat["season"] = season
    df_lg_batting_stat["league"] = league
    df_lg_batting_stat.set_index("Stat", inplace=True)

    pd.DataFrame(league_batting_totals, index=[0]).to_csv(
        f"./files/{league}/{season}/output/{league}-{season}-batting-totals.csv"
    )
    df_lg_batting_stat.round(9).to_csv(
        f"./files/{league}/{season}/output/{league}-{season}-woba-calcs.csv"
    )

    (
        df_player_batting_stats,
        df_player_batting_stats_right,
        df_player_batting_stats_left,
    ) = calculate_player_batting_stats(
        df_player_batting_stats,
        df_player_batting_right_stats,
        df_player_batting_left_stats,
        df_lg_batting_stat,
        df_player_ratings,
        league,
    )
    df_player_batting_stats["season"] = season
    df_player_batting_stats["league"] = league
    df_player_batting_stats.to_csv(
        f"./files/{league}/{season}/output/{league}-{season}-hitting.csv"
    )
    df_player_batting_stats_right.to_csv(
        f"./files/{league}/{season}/output/{league}-{season}-right-hitting.csv"
    )
    df_player_batting_stats_left.to_csv(
        f"./files/{league}/{season}/output/{league}-{season}-left-hitting.csv"
    )

    return (
        df_player_batting_stats,
        df_lg_batting_stat,
        df_player_batting_stats_right,
        df_player_batting_stats_left,
    )
