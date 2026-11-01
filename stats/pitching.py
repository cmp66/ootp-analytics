import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from util import paths
from util.innings import ip_to_innings
from stats import woba

RUNS_PER_WIN = 10
PITCHER_ROLES = ["SP", "RP", "Mix"]


def classify_pitcher_role(df_player_stats: DataFrame) -> Series:
    """SP: every appearance was a start. RP: never started. Mix: anything in between."""
    return Series(
        np.select(
            [df_player_stats["GS"] == df_player_stats["G"], df_player_stats["GS"] == 0],
            ["SP", "RP"],
            default="Mix",
        ),
        index=df_player_stats.index,
    )


def calculate_league_totals(df_player_stats: DataFrame) -> DataFrame:
    """IP, runs and RA9 by pitcher role (SP, RP, Mix) plus the whole league (Total).

    Each role is compared with its own RA9, so WAA sums to about 0 within every role.
    A role nobody pitched as is NaN."""
    by_role = df_player_stats.groupby("PRole")[["IPClean", "R"]].sum()
    df_pitching_totals = by_role.reindex(PITCHER_ROLES)
    df_pitching_totals.loc["Total"] = df_player_stats[["IPClean", "R"]].sum()
    df_pitching_totals.index.name = "POS"
    df_pitching_totals["RA9"] = (
        df_pitching_totals["R"] / df_pitching_totals["IPClean"] * 9
    )
    return df_pitching_totals


def calculate_pitching_attribute_averages(
    df_player_pitching_stats: DataFrame,
    df_player_ratings: DataFrame,
    df_lg_pitching_stat: DataFrame,
) -> DataFrame:

    df_player_stats_temp = df_player_pitching_stats.copy()
    df_player_stats_temp = df_player_stats_temp.merge(df_player_ratings, on="ID")
    # weight only by the batters faced of pitchers that have ratings
    lg_bf = df_player_stats_temp["BF"].sum()

    df_player_stats_temp["pbabip_weight"] = (
        df_player_stats_temp["PBABIP"] * df_player_stats_temp["BF"]
    )
    df_player_stats_temp["stu_weight"] = (
        df_player_stats_temp["STU"] * df_player_stats_temp["BF"]
    )
    df_player_stats_temp["hrr_weight"] = (
        df_player_stats_temp["HRR"] * df_player_stats_temp["BF"]
    )
    df_player_stats_temp["ctl_weight"] = (
        df_player_stats_temp["CON.1"] * df_player_stats_temp["BF"]
    )
    df_player_stats_temp["mov_weight"] = (
        df_player_stats_temp["MOV"] * df_player_stats_temp["BF"]
    )

    df_lg_pitching_stat.loc["lgPbabipRating"] = (
        df_player_stats_temp["pbabip_weight"].sum() / lg_bf
    )
    df_lg_pitching_stat.loc["lgStuRating"] = (
        df_player_stats_temp["stu_weight"].sum() / lg_bf
    )
    df_lg_pitching_stat.loc["lgHrrRating"] = (
        df_player_stats_temp["hrr_weight"].sum() / lg_bf
    )
    df_lg_pitching_stat.loc["lgCtlRating"] = (
        df_player_stats_temp["ctl_weight"].sum() / lg_bf
    )
    df_lg_pitching_stat.loc["lgMovRating"] = (
        df_player_stats_temp["mov_weight"].sum() / lg_bf
    )

    return df_lg_pitching_stat


def calculate_player_pitching_stats(
    df_player_stats: DataFrame,
    df_lg_stat: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
    df_lg_woba: DataFrame | None = None,
) -> DataFrame:
    """df_lg_woba holds the wOBA coefficients / run values; defaults to the pitching-side league stats."""

    lg = (df_lg_stat if df_lg_woba is None else df_lg_woba)["Value"]
    df_player_stats = df_player_stats.merge(df_player_ratings, on="ID")

    df_player_stats["PRole"] = classify_pitcher_role(df_player_stats)
    df_player_stats["IPClean"] = ip_to_innings(df_player_stats["IP"])
    df_player_stats["RA9"] = df_player_stats["R"] / df_player_stats["IPClean"] * 9
    df_pitching_totals = calculate_league_totals(df_player_stats)

    lg_ra9 = df_player_stats["PRole"].map(df_pitching_totals["RA9"])
    df_player_stats["wSBA"] = (
        df_player_stats["SB"] * lg["run_value_sb"]
        + df_player_stats["CS"] * lg["run_value_cs"]
    )
    df_player_stats["WAA"] = (
        lg_ra9 * df_player_stats["IPClean"] / 9
        - df_player_stats["IPClean"] * df_player_stats["RA9"] / 9
    ) / RUNS_PER_WIN
    df_player_stats["KPercent"] = df_player_stats["K"] / df_player_stats["BF"] * 100
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
    # same formula and coefficients as batter wOBA
    df_player_stats["wOBAa"] = woba.calc_woba(df_player_stats, lg)
    df_player_stats["WAA200"] = (
        df_player_stats["WAA"] * 200 / df_player_stats["IPClean"]
    )

    # Park factors removed; kept as an alias of WAA for downstream consumers
    df_player_stats["WAAAdj"] = df_player_stats["WAA"]

    columns_to_remove = [
        x for x in df_player_ratings.columns if x not in ["ID", "POS", "Name"]
    ]
    df_player_stats = df_player_stats.drop(columns=columns_to_remove)
    df_player_stats = df_player_stats.assign(**df_lg_stat["Value"].to_dict())

    return df_player_stats.set_index("ID")


def calculate_league_pitching_totals(
    df_player_pitching_stats: DataFrame,
) -> dict[str, float]:
    return {
        "PA": df_player_pitching_stats["BF"].sum(),
        "AB": df_player_pitching_stats["AB"].sum(),
        "R": df_player_pitching_stats["R"].sum(),
        "H": df_player_pitching_stats["HA"].sum(),
        "1B": df_player_pitching_stats["SINGLE"].sum(),
        "2B": df_player_pitching_stats["DOUBLE"].sum(),
        "3B": df_player_pitching_stats["TRIPLE"].sum(),
        "HR": df_player_pitching_stats["HR"].sum(),
        "BB": df_player_pitching_stats["BB"].sum(),
        "IBB": df_player_pitching_stats["IBB"].sum(),
        "HP": df_player_pitching_stats["HP"].sum(),
        "SO": df_player_pitching_stats["K"].sum(),
        "SF": df_player_pitching_stats["SF"].sum(),
        "SH": df_player_pitching_stats["SH"].sum(),
        "GIDP": df_player_pitching_stats["DP"].sum(),
        "SB": df_player_pitching_stats["SB"].sum(),
        "CS": df_player_pitching_stats["CS"].sum(),
    }


def process_pitching_data(
    df_player_pitching_stats: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
    season: int,
    df_lg_batting_stat: DataFrame | None = None,
) -> DataFrame:
    """df_lg_batting_stat is the league table returned by batting.process_batting_data.
    Its wOBA coefficients are used for wOBAa so batters and pitchers share one set;
    when omitted it is read from the woba-calcs csv written by the batting step."""
    league_pitching_totals = calculate_league_pitching_totals(df_player_pitching_stats)

    df_lg_pitching_stat = DataFrame(
        list(woba.calc_league_data(league_pitching_totals).items()),
        columns=["Stat", "Value"],
    )
    df_lg_pitching_stat["season"] = season
    df_lg_pitching_stat["league"] = league
    df_lg_pitching_stat = df_lg_pitching_stat.set_index("Stat")

    if df_lg_batting_stat is None:
        df_lg_batting_stat = pd.read_csv(
            paths.output_file(league, season, "woba-calcs")
        ).set_index("Stat")

    df_player_pitching_stats = calculate_player_pitching_stats(
        df_player_pitching_stats,
        df_lg_pitching_stat,
        df_player_ratings,
        league,
        df_lg_woba=df_lg_batting_stat,
    )

    df_lg_pitching_stat = calculate_pitching_attribute_averages(
        df_player_pitching_stats,
        df_player_ratings,
        df_lg_pitching_stat,
    )
    df_player_pitching_stats["season"] = season
    df_player_pitching_stats["league"] = league

    df_woba = df_lg_batting_stat.copy()
    df_woba.loc["lgBF"] = league_pitching_totals["PA"], season, league
    df_woba.loc["lgPbabipRating"] = (
        df_lg_pitching_stat.loc["lgPbabipRating"]["Value"],
        season,
        league,
    )
    df_woba.loc["lgStuRating"] = (
        df_lg_pitching_stat.loc["lgStuRating"]["Value"],
        season,
        league,
    )
    df_woba.loc["lgHrrRating"] = (
        df_lg_pitching_stat.loc["lgHrrRating"]["Value"],
        season,
        league,
    )
    df_woba.loc["lgCtlRating"] = (
        df_lg_pitching_stat.loc["lgCtlRating"]["Value"],
        season,
        league,
    )
    df_woba.loc["lgMovRating"] = (
        df_lg_pitching_stat.loc["lgMovRating"]["Value"],
        season,
        league,
    )
    df_woba.to_csv(paths.output_file(league, season, "woba-calcs"))

    DataFrame(league_pitching_totals, index=[0]).to_csv(
        paths.output_file(league, season, "pitching-totals")
    )
    df_player_pitching_stats.to_csv(paths.output_file(league, season, "pitching"))

    return df_player_pitching_stats
