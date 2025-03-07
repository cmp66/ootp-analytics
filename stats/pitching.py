from pandas import DataFrame
import math
from stats import leagueAdjustments


def calculate_league_totals(df_player_stats: DataFrame) -> dict[str, float]:

    pitcher_group = df_player_stats.groupby("PRole")
    df_pitching_totals = DataFrame({"POS": ["SP", "RP", "Mix", "Total"]})
    df_pitching_totals.set_index("POS", inplace=True)

    df_pitching_totals.loc["SP", "IPClean"] = pitcher_group["IPClean"].sum().loc["SP"]
    df_pitching_totals.loc["SP", "R"] = pitcher_group["R"].sum().loc["SP"]
    df_pitching_totals.loc["SP", "RA9"] = (
        df_pitching_totals.loc["SP"]["R"] / df_pitching_totals.loc["SP"]["IPClean"] * 9
    )

    df_pitching_totals.loc["RP", "IPClean"] = pitcher_group["IPClean"].sum().loc["RP"]
    df_pitching_totals.loc["RP", "R"] = pitcher_group["R"].sum().loc["RP"]
    df_pitching_totals.loc["RP", "RA9"] = (
        df_pitching_totals.loc["RP"]["R"] / df_pitching_totals.loc["RP"]["IPClean"] * 9
    )

    df_pitching_totals.loc["Mix", "IPClean"] = df_player_stats["IPClean"].sum()
    df_pitching_totals.loc["Mix", "R"] = df_player_stats["R"].sum()
    df_pitching_totals.loc["Mix", "RA9"] = (
        df_pitching_totals.loc["Mix"]["R"]
        / df_pitching_totals.loc["Mix"]["IPClean"]
        * 9
    )

    df_pitching_totals.loc["Total", "IPClean"] = df_player_stats["IPClean"].sum()
    df_pitching_totals.loc["Total", "R"] = df_player_stats["R"].sum()
    df_pitching_totals.loc["Total", "RA9"] = (
        df_pitching_totals.loc["Total"]["R"]
        / df_pitching_totals.loc["Total"]["IPClean"]
        * 9
    )

    return df_pitching_totals


def calculate_player_pitching_stats(
    df_player_stats: DataFrame,
    df_lg_stat: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
) -> DataFrame:

    df_player_stats = df_player_stats.merge(df_player_ratings, on="ID")
    park_adjustments = leagueAdjustments.get_park_adjustments(league)

    df_player_stats["PRole"] = df_player_stats[["G", "GS"]].apply(
        lambda x: "SP" if x["GS"] == x["G"] else "RP" if x["GS"] == 0 else "Mix", axis=1
    )
    df_player_stats["IPClean"] = df_player_stats["IP"].apply(
        lambda x: math.modf(x)[1] + math.modf(x)[0] * 0.33
    )
    df_player_stats["RA9"] = df_player_stats["R"] / df_player_stats["IPClean"] * 9
    df_pitching_totals = calculate_league_totals(df_player_stats)

    df_player_stats["wSBA"] = (
        df_player_stats["SB"] * df_lg_stat.loc["run_value_sb"]["Value"]
        + df_player_stats["CS"] * df_lg_stat.loc["run_value_cs"]["Value"]
    )
    df_player_stats["WAA"] = (
        df_player_stats["PRole"].apply(lambda x: df_pitching_totals.loc[x, "RA9"])
        * df_player_stats["IPClean"]
        / 9
        - df_player_stats["IPClean"] * df_player_stats["RA9"] / 9
    ) / 10
    df_player_stats["KPercent"] = (
        df_player_stats["K"]
        / (
            df_player_stats["BF"]
            - df_player_stats["BB"]
            - df_player_stats["HR"]
            - df_player_stats["HP"]
        )
        * 100
    )
    df_player_stats["HRPercent"] = df_player_stats["HR"] / df_player_stats["BF"] * 100
    df_player_stats["pBABIP"] = (
        df_player_stats["SINGLE"]
        + df_player_stats["DOUBLE"]
        + df_player_stats["TRIPLE"]
    ) / (
        df_player_stats["AB"]
        - df_player_stats["K"]
        - df_player_stats["HR"]
        + df_player_stats["SF"]
    )
    df_player_stats["BBPercent"] = df_player_stats["BB"] / df_player_stats["BF"] * 100
    df_player_stats["wOBAa"] = (
        df_player_stats["HP"] * df_lg_stat.loc["coef_HP"]["Value"]
        + df_player_stats["BB"] * df_lg_stat.loc["coef_BB"]["Value"]
        + df_player_stats["SINGLE"] * df_lg_stat.loc["coef_1B"]["Value"]
        + df_player_stats["DOUBLE"] * df_lg_stat.loc["coef_2B"]["Value"]
        + df_player_stats["TRIPLE"] * df_lg_stat.loc["coef_3B"]["Value"]
        + df_player_stats["HR"] * df_lg_stat.loc["coef_HR"]["Value"]
        + df_player_stats["SB"] * df_lg_stat.loc["coef_SB"]["Value"]
        + df_player_stats["CS"] * df_lg_stat.loc["coef_CS"]["Value"]
    ) / df_player_stats["BF"]
    df_player_stats["WAA200"] = (
        df_player_stats["WAA"] * 200 / df_player_stats["IPClean"]
    )
    df_player_stats["WAAAdj"] = (
        (
            df_player_stats["PRole"].apply(lambda x: df_pitching_totals.loc[x, "RA9"])
            * df_player_stats["IPClean"]
            / 9
        )
        - (
            df_player_stats["IPClean"] * df_player_stats["RA9"] / 9
            + df_player_stats["ORG"].apply(
                lambda x: park_adjustments[x] if x in park_adjustments else 0
            )
            / 600
            * df_player_stats["BF"]
        )
    ) / 10

    columns_to_remove = [
        x for x in df_player_ratings.columns if x not in ["ID", "POS", "Name"]
    ]
    df_player_stats.drop(columns_to_remove, axis=1, inplace=True)

    df_player_stats.set_index("ID", inplace=True)

    return df_player_stats
