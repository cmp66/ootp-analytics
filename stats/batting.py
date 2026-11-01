from pandas import DataFrame, Series
import pandas as pd
from util import paths
from stats import woba


def calculate_batting_attribute_averages(
    df_player_batting_stats: DataFrame,
    df_player_ratings: DataFrame,
    df_lg_batting_stat: DataFrame,
) -> DataFrame:

    df_player_stats_temp = df_player_batting_stats.copy()
    df_player_stats_temp = df_player_stats_temp.merge(df_player_ratings, on="ID")
    # weight only by the PA of players that have ratings
    lg_pa = df_player_stats_temp["PA"].sum()

    df_player_stats_temp["babip_weight"] = (
        df_player_stats_temp["BABIP"] * df_player_stats_temp["PA"]
    )
    df_player_stats_temp["avk_weight"] = (
        df_player_stats_temp["K's"] * df_player_stats_temp["PA"]
    )
    df_player_stats_temp["pow_weight"] = (
        df_player_stats_temp["POW"] * df_player_stats_temp["PA"]
    )
    df_player_stats_temp["con_weight"] = (
        df_player_stats_temp["CON"] * df_player_stats_temp["PA"]
    )
    df_player_stats_temp["eye_weight"] = (
        df_player_stats_temp["EYE"] * df_player_stats_temp["PA"]
    )

    df_lg_batting_stat.loc["lgBabipRating"] = (
        df_player_stats_temp["babip_weight"].sum() / lg_pa
    )
    df_lg_batting_stat.loc["lgAvkRating"] = (
        df_player_stats_temp["avk_weight"].sum() / lg_pa
    )
    df_lg_batting_stat.loc["lgConRating"] = (
        df_player_stats_temp["con_weight"].sum() / lg_pa
    )
    df_lg_batting_stat.loc["lgEyeRating"] = (
        df_player_stats_temp["eye_weight"].sum() / lg_pa
    )
    df_lg_batting_stat.loc["lgPowRating"] = (
        df_player_stats_temp["pow_weight"].sum() / lg_pa
    )

    return df_lg_batting_stat


def add_split_batting_stats(
    df_split: DataFrame, df_lg_stat: DataFrame, df_bsr: Series, suffix: str
) -> DataFrame:
    """wOBA / wRAA / OFF columns for a vs-left or vs-right batting split, joined to BSR by player ID."""
    lg = df_lg_stat["Value"]
    df_split = df_split.copy()
    df_split[f"wOBA{suffix}"] = woba.calc_woba(df_split, lg)
    df_split[f"wRAA{suffix}"] = (
        (df_split[f"wOBA{suffix}"] - lg["lgwOBA"]) / lg["wOBA_SCALE"]
    ) * woba.woba_denominator(df_split)
    df_split[f"wRAA600{suffix}"] = df_split[f"wRAA{suffix}"] * 600 / df_split["PA"]
    df_split[f"OFF{suffix}"] = df_split["ID"].map(df_bsr) + df_split[f"wRAA{suffix}"]
    df_split[f"OFF600{suffix}"] = df_split[f"OFF{suffix}"] * 600 / df_split["PA"]
    return df_split


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
    lg = df_lg_stat["Value"]

    df_player_stats_total = df_player_stats_total.merge(df_player_ratings, on="ID")

    if not skip_handedness:
        df_player_stats_right = df_player_stats_right.merge(df_player_ratings, on="ID")
        df_player_stats_left = df_player_stats_left.merge(df_player_ratings, on="ID")

    df_total = df_player_stats_total
    df_total["UBRAA"] = (
        df_total["UBR"]
        - (
            (df_total["SINGLE"] + df_total["BB"] + df_total["HP"]) * 3
            + df_total["DOUBLE"] * 2
            + df_total["TRIPLE"]
            - df_total["SB"]
            - df_total["CS"] * 3
        )
        * lg["lgUBR"]
    )

    df_total["wOBA"] = woba.calc_woba(df_total, lg)
    df_total["wSB"] = (
        (df_total["SB"] * lg["run_value_sb"])
        + (df_total["CS"] * lg["run_value_cs"])
        - (
            lg["lgwSB"]
            * (df_total["SINGLE"] + df_total["BB"] + df_total["HP"] - df_total["IBB"])
        )
    )
    df_total["wRAA"] = (
        (df_total["wOBA"] - lg["lgwOBA"]) / lg["wOBA_SCALE"]
    ) * woba.woba_denominator(df_total)
    df_total["wRAA600"] = df_total["wRAA"] * 600 / df_total["PA"]
    df_total["OBP"] = (df_total["H"] + df_total["BB"] + df_total["HP"]) / (
        df_total["AB"] + df_total["BB"] + df_total["HP"] + df_total["SF"]
    )
    df_total["BSR"] = df_total["UBRAA"] + df_total["wSB"]
    df_total["BSR600"] = df_total["BSR"] * 600 / df_total["PA"]
    df_total["OFF"] = df_total["BSR"] + df_total["wRAA"]
    # Park factors removed; kept as an alias of OFF for downstream consumers
    df_total["OFFAdj"] = df_total["OFF"]
    df_total["OFF600"] = df_total["OFF"] * 600 / df_total["PA"]
    df_total["SBAPERCENT"] = (df_total["SB"] + df_total["CS"]) / (
        df_total["SINGLE"] + df_total["BB"] + df_total["HP"]
    )

    if not skip_handedness:
        df_bsr = df_total.set_index("ID")["BSR"]
        df_player_stats_right = add_split_batting_stats(
            df_player_stats_right, df_lg_stat, df_bsr, "Right"
        )
        df_player_stats_left = add_split_batting_stats(
            df_player_stats_left, df_lg_stat, df_bsr, "Left"
        )

    columns_to_remove = [
        x for x in df_player_ratings.columns if x not in (["ID", "POS", "Name"])
    ]
    lg_columns = lg.to_dict()

    def finish(df: DataFrame) -> DataFrame:
        df = df.drop(columns=columns_to_remove)
        df = df.assign(**lg_columns)
        return df.set_index("ID")

    df_player_stats_total = finish(df_total)
    if not skip_handedness:
        df_player_stats_right = finish(df_player_stats_right)
        df_player_stats_left = finish(df_player_stats_left)

    return df_player_stats_total, df_player_stats_right, df_player_stats_left


def calculate_league_batting_totals(
    df_player_batting_stats: DataFrame, source: str
) -> dict[str, float]:
    """League counting-stat totals. Export PA excludes SH, SF and CI, so they are added back."""
    return {
        "PA": (
            df_player_batting_stats["PA"].sum()
            if source == "dump"
            else df_player_batting_stats["PA"].sum()
            + df_player_batting_stats["SH"].sum()
            + df_player_batting_stats["SF"].sum()
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


def process_batting_data(
    df_player_batting_stats: DataFrame,
    df_player_batting_right_stats: DataFrame,
    df_player_batting_left_stats: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
    season: int,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    league_batting_totals = calculate_league_batting_totals(
        df_player_batting_stats, source
    )

    df_lg_batting_stat = pd.DataFrame(
        list(woba.calc_league_data(league_batting_totals).items()),
        columns=["Stat", "Value"],
    )
    df_lg_batting_stat = df_lg_batting_stat.set_index("Stat")

    pd.DataFrame(league_batting_totals, index=[0]).to_csv(
        paths.output_file(league, season, "batting-totals")
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
    for df_split in (df_player_batting_stats_right, df_player_batting_stats_left):
        if not df_split.empty:
            df_split["season"] = season
            df_split["league"] = league
    df_player_batting_stats.to_csv(paths.output_file(league, season, "hitting"))
    df_player_batting_stats_right.to_csv(
        paths.output_file(league, season, "right-hitting")
    )
    df_player_batting_stats_left.to_csv(
        paths.output_file(league, season, "left-hitting")
    )

    df_lg_batting_stat = calculate_batting_attribute_averages(
        df_player_batting_stats,
        df_player_ratings,
        df_lg_batting_stat,
    )
    df_lg_batting_stat["season"] = season
    df_lg_batting_stat["league"] = league
    df_lg_batting_stat.round(9).to_csv(paths.output_file(league, season, "woba-calcs"))

    return (
        df_player_batting_stats,
        df_lg_batting_stat,
        df_player_batting_stats_right,
        df_player_batting_stats_left,
    )
