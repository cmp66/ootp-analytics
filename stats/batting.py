from pandas import DataFrame, Series
import pandas as pd
from util import paths
from stats import parks, woba


def calculate_batting_attribute_averages(
    df_player_batting_stats: DataFrame,
    df_player_ratings: DataFrame,
    df_lg_batting_stat: DataFrame,
) -> DataFrame:

    df_player_stats_temp = df_player_batting_stats.copy()
    df_player_stats_temp = df_player_stats_temp.merge(df_player_ratings, on="ID")
    # weight only by the PA of players that have ratings
    lg_pa = df_player_stats_temp["PA"].sum()

    df_player_stats_temp["babip_weight"] = df_player_stats_temp["BABIP"] * df_player_stats_temp["PA"]
    df_player_stats_temp["avk_weight"] = df_player_stats_temp["K's"] * df_player_stats_temp["PA"]
    df_player_stats_temp["pow_weight"] = df_player_stats_temp["POW"] * df_player_stats_temp["PA"]
    df_player_stats_temp["con_weight"] = df_player_stats_temp["CON"] * df_player_stats_temp["PA"]
    df_player_stats_temp["eye_weight"] = df_player_stats_temp["EYE"] * df_player_stats_temp["PA"]

    df_lg_batting_stat.loc["lgBabipRating"] = df_player_stats_temp["babip_weight"].sum() / lg_pa
    df_lg_batting_stat.loc["lgAvkRating"] = df_player_stats_temp["avk_weight"].sum() / lg_pa
    df_lg_batting_stat.loc["lgConRating"] = df_player_stats_temp["con_weight"].sum() / lg_pa
    df_lg_batting_stat.loc["lgEyeRating"] = df_player_stats_temp["eye_weight"].sum() / lg_pa
    df_lg_batting_stat.loc["lgPowRating"] = df_player_stats_temp["pow_weight"].sum() / lg_pa

    return df_lg_batting_stat


def add_park_adjusted_offense(
    df_total: DataFrame,
    lg: pd.Series,
    df_park_factors: DataFrame | None,
    pa_vs_right: pd.Series | None = None,
    pa_vs_left: pd.Series | None = None,
) -> None:
    """Add ParkAVG / ParkHR / ParkD / ParkT, wOBAAdj, wRAAAdj and OFFAdj to df_total (in place).

    Each batted-ball event is divided by the park factor the batter faced (hits by AVG, doubles by D, triples
    by T, home runs by HR; left or right column by the side he bats from), singles are what remains of the
    adjusted hits. Walks, hit by pitch and strikeouts are untouched, and AB stays the same (a hit that does not
    happen is an out). The baseline is the league mean of the adjusted wOBA, so league wRAAAdj sums to 0.
    Without park factors every factor is 1 and OFFAdj equals OFF."""
    ids = df_total["ID"]
    if df_park_factors is None:
        factors = pd.DataFrame(1.0, index=df_total.index, columns=parks.BATTING_FACTORS)
    else:
        factors = df_park_factors.reindex(ids).fillna(1.0).set_axis(df_total.index)

    # share of a batter's PA taken from the left side: lefties 1, righties 0, switch hitters by pitcher hand faced
    bats = df_total["B"]
    left_share = pd.Series(0.5, index=df_total.index)
    left_share[bats == 1] = 0.0
    left_share[bats == 2] = 1.0
    if pa_vs_right is not None and pa_vs_left is not None:
        vs_right = ids.map(pa_vs_right).fillna(0)
        vs_left = ids.map(pa_vs_left).fillna(0)
        switch = (bats == 3) & (vs_right + vs_left > 0)
        left_share[switch] = (vs_right / (vs_right + vs_left))[switch]

    f_avg = left_share * factors["AVG_L"] + (1 - left_share) * factors["AVG_R"]
    f_hr = left_share * factors["HR_L"] + (1 - left_share) * factors["HR_R"]
    homers = df_total["HR"] / f_hr
    triples = df_total["TRIPLE"] / factors["T"]
    doubles = df_total["DOUBLE"] / factors["D"]
    singles = (df_total["H"] / f_avg - doubles - triples - homers).clip(lower=0)

    numerator = (
        singles * lg["coef_1B"]
        + doubles * lg["coef_2B"]
        + triples * lg["coef_3B"]
        + homers * lg["coef_HR"]
        + (df_total["BB"] - df_total["IBB"]) * lg["coef_BB"]
        + df_total["HP"] * lg["coef_HP"]
    )
    denominator = woba.woba_denominator(df_total)
    woba_adj = numerator / denominator
    lg_woba_adj = numerator.sum() / denominator.sum()

    df_total["ParkAVG"] = f_avg
    df_total["ParkHR"] = f_hr
    df_total["ParkD"] = factors["D"]
    df_total["ParkT"] = factors["T"]
    df_total["wOBAAdj"] = woba_adj
    df_total["wRAAAdj"] = (woba_adj - lg_woba_adj) / lg["wOBA_SCALE"] * denominator
    df_total["OFFAdj"] = df_total["BSR"] + df_total["wRAAAdj"]


def add_split_batting_stats(df_split: DataFrame, df_lg_stat: DataFrame, df_bsr: Series, suffix: str) -> DataFrame:
    """wOBA / wRAA / OFF columns for a vs-left or vs-right batting split, joined to BSR by player ID."""
    lg = df_lg_stat["Value"]
    df_split = df_split.copy()
    df_split[f"wOBA{suffix}"] = woba.calc_woba(df_split, lg)
    df_split[f"wRAA{suffix}"] = ((df_split[f"wOBA{suffix}"] - lg["lgwOBA"]) / lg["wOBA_SCALE"]) * woba.woba_denominator(
        df_split
    )
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
    df_park_factors: DataFrame | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """df_park_factors: effective park factors per player ID (stats.parks.ParkFactors.batting); neutral when omitted."""

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
        - (lg["lgwSB"] * (df_total["SINGLE"] + df_total["BB"] + df_total["HP"] - df_total["IBB"]))
    )
    df_total["wRAA"] = ((df_total["wOBA"] - lg["lgwOBA"]) / lg["wOBA_SCALE"]) * woba.woba_denominator(df_total)
    df_total["wRAA600"] = df_total["wRAA"] * 600 / df_total["PA"]
    df_total["OBP"] = (df_total["H"] + df_total["BB"] + df_total["HP"]) / (
        df_total["AB"] + df_total["BB"] + df_total["HP"] + df_total["SF"]
    )
    df_total["BSR"] = df_total["UBRAA"] + df_total["wSB"]
    df_total["BSR600"] = df_total["BSR"] * 600 / df_total["PA"]
    df_total["OFF"] = df_total["BSR"] + df_total["wRAA"]
    pa_vs_right = None if skip_handedness else df_player_stats_right.set_index("ID")["PA"]
    pa_vs_left = None if skip_handedness else df_player_stats_left.set_index("ID")["PA"]
    add_park_adjusted_offense(df_total, lg, df_park_factors, pa_vs_right, pa_vs_left)
    df_total["OFF600"] = df_total["OFF"] * 600 / df_total["PA"]
    df_total["SBAPERCENT"] = (df_total["SB"] + df_total["CS"]) / (df_total["SINGLE"] + df_total["BB"] + df_total["HP"])

    if not skip_handedness:
        df_bsr = df_total.set_index("ID")["BSR"]
        df_player_stats_right = add_split_batting_stats(df_player_stats_right, df_lg_stat, df_bsr, "Right")
        df_player_stats_left = add_split_batting_stats(df_player_stats_left, df_lg_stat, df_bsr, "Left")

    columns_to_remove = [x for x in df_player_ratings.columns if x not in (["ID", "POS", "Name"])]
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


def calculate_league_batting_totals(df_player_batting_stats: DataFrame, source: str) -> dict[str, float]:
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
    df_park_factors: DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    league_batting_totals = calculate_league_batting_totals(df_player_batting_stats, source)

    df_lg_batting_stat = pd.DataFrame(
        list(woba.calc_league_data(league_batting_totals).items()),
        columns=["Stat", "Value"],
    )
    df_lg_batting_stat = df_lg_batting_stat.set_index("Stat")

    pd.DataFrame(league_batting_totals, index=[0]).to_csv(paths.output_file(league, season, "batting-totals"))

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
        df_park_factors,
    )
    df_player_batting_stats["season"] = season
    df_player_batting_stats["league"] = league
    for df_split in (df_player_batting_stats_right, df_player_batting_stats_left):
        if not df_split.empty:
            df_split["season"] = season
            df_split["league"] = league
    df_player_batting_stats.to_csv(paths.output_file(league, season, "hitting"))
    df_player_batting_stats_right.to_csv(paths.output_file(league, season, "right-hitting"))
    df_player_batting_stats_left.to_csv(paths.output_file(league, season, "left-hitting"))

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
