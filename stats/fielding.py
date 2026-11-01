from util import paths
import numpy as np
from pandas import DataFrame, Index, Series
from util.innings import ip_to_innings

positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
of_positions = [7, 8, 9]
if_positions = [3, 4, 5, 6]
dp_positions = [4, 6]


def get_season_adjustment(pos: int) -> int:
    """Innings in a full season at a position (catchers rest more often)."""
    return 220 if pos == 1 else 1100 if pos == 2 else 1200


# one season length for every calculation
get_fielding_adjustment = get_season_adjustment


# Double plays are credited for participation, not skill (see plans/05-fielding.md), so the DPAA
# columns are kept for reference but are not part of any fielding runs total.
def get_dp_adjustment(pos: int) -> float:
    """Share of double-play credit given to each position (2B pivots, SS feeds; nobody else is credited)."""
    return 0.6 if pos in [4] else 0.4 if pos in [6] else 0.0


def get_out_run_value(pos: str) -> float:
    return 0.9 if pos in [7, 8, 9] else 0.75


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


def _by_position(func, pos: Series) -> Series:
    return pos.map({p: func(p) for p in pos.unique()})


def calculate_league_totals(
    df_player_stats: DataFrame,
) -> tuple[DataFrame, DataFrame]:

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

    total_columns = [
        "PlaysMade",
        "PlaysAttempted",
        "IPClean",
        "E",
        "ARM",
        "SBA",
        "RTO",
        "FRM",
        "DP",
    ]
    sums = df_player_stats.groupby("POS")[total_columns].sum().reindex(positions)
    season = Series({p: get_season_adjustment(p) for p in positions})

    df_positional_totals = DataFrame(index=Index(positions, name="POS"))
    df_positional_totals["PlayPercent"] = sums["PlaysMade"] / sums["PlaysAttempted"] * 100
    df_positional_totals["PlayAttemptedPerSeason"] = sums["PlaysAttempted"] / sums["IPClean"] * season
    df_positional_totals["PlayPerSeason"] = sums["PlaysMade"] / sums["IPClean"] * season
    df_positional_totals["ErrorPerSeason"] = sums["E"] / sums["IPClean"] * season
    df_positional_totals["ErrorPercentage"] = sums["E"] / sums["PlaysMade"] * 100

    is_of = df_positional_totals.index.isin(of_positions)
    df_positional_totals["ARM"] = (
        sums["ARM"] / sums["PlaysAttempted"] * df_positional_totals["PlayAttemptedPerSeason"]
    ).where(is_of, 0)

    is_catcher = df_positional_totals.index == 2
    df_positional_totals["SBA"] = (sums["SBA"] / sums["IPClean"] * 1000).where(is_catcher, 0)
    df_positional_totals["RTO"] = (sums["RTO"] / sums["SBA"]).where(is_catcher, 0)
    df_positional_totals["FRM"] = (sums["FRM"] / sums["IPClean"]).where(is_catcher, 0)

    is_if = df_positional_totals.index.isin(if_positions)
    df_positional_totals["DP"] = (sums["DP"] / sums["IPClean"] * season).where(is_if, 0)

    df_lg_fielding_stat = DataFrame(list(league_fielding_totals.items()), columns=["Stat", "Value"]).set_index("Stat")

    return df_lg_fielding_stat, df_positional_totals


# Rating column that holds the player's rating at each fielding position
POSITION_RATING_COLUMN = {
    1: "P",
    2: "C",
    3: "1B",
    4: "2B",
    5: "3B",
    6: "SS",
    7: "LF",
    8: "CF",
    9: "RF",
}


def calculate_league_attribute_averages(df_player_stats: DataFrame) -> DataFrame:
    """Innings-weighted average rating of the players who fielded each position."""

    def weighted_mean(column: str) -> Series:
        weights = df_player_stats["IPClean"] / df_player_stats.groupby("POS")["IPClean"].transform("sum")
        return (weights * column_values(column)).groupby(df_player_stats["POS"]).sum()

    def column_values(column: str) -> Series:
        if column == "POSITION":
            # each player's rating at the position they fielded
            ratings = df_player_stats[[POSITION_RATING_COLUMN[p] for p in positions]].to_numpy()
            return Series(
                ratings[np.arange(len(ratings)), df_player_stats["POS"].to_numpy() - 1],
                index=df_player_stats.index,
            )
        return df_player_stats[column]

    df_fielding_averages = DataFrame(index=Index(positions, name="POS"))
    in_of = df_fielding_averages.index.isin(of_positions)
    in_if = df_fielding_averages.index.isin(if_positions + [1])
    is_catcher = df_fielding_averages.index == 2

    def pick(of_column: str, if_column: str, catcher_column: str | None = None) -> Series:
        result = Series(float("nan"), index=df_fielding_averages.index)
        result[in_of] = weighted_mean(of_column).reindex(result.index)[in_of]
        result[in_if] = weighted_mean(if_column).reindex(result.index)[in_if]
        if catcher_column:
            result[is_catcher] = weighted_mean(catcher_column).reindex(result.index)[is_catcher]
        return result

    df_fielding_averages["PRat"] = pick("POSITION", "POSITION", "POSITION")
    df_fielding_averages["Range"] = pick("OF RNG", "IF RNG")
    df_fielding_averages["Error"] = pick("OF ERR", "IF ERR")
    df_fielding_averages["Arm"] = pick("OF ARM", "IF ARM", "C ARM")
    df_fielding_averages["DP"] = float("nan")
    df_fielding_averages.loc[in_of, "DP"] = 0.0
    df_fielding_averages.loc[in_if, "DP"] = weighted_mean("TDP").reindex(df_fielding_averages.index)[in_if]
    df_fielding_averages["Frame"] = float("nan")
    df_fielding_averages["Block"] = float("nan")
    df_fielding_averages.loc[2, "Frame"] = weighted_mean("C FRM").get(2, 0.0)
    df_fielding_averages.loc[2, "Block"] = weighted_mean("C ABI").get(2, 0.0)

    return df_fielding_averages.round(2)


def calculate_player_fielding_stats(
    df_player_stats: DataFrame, df_lg_stat: DataFrame, df_player_ratings: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:

    # DH and any other non-fielding rows have no league baseline
    df_player_stats = df_player_stats[df_player_stats["POS"].isin(positions)].copy()
    pos = df_player_stats["POS"]

    df_player_stats["DP"] = df_player_stats["DP"] * _by_position(get_dp_adjustment, pos)
    df_player_stats["ARM"] = df_player_stats["ARM"] * pos.isin(of_positions)

    df_player_stats = df_player_stats.merge(df_player_ratings, on="ID")
    pos = df_player_stats["POS"]

    df_player_stats["PlaysAttempted"] = (
        df_player_stats["BIZ-R"]
        + df_player_stats["BIZ-L"]
        + df_player_stats["BIZ-E"]
        + df_player_stats["BIZ-U"]
        + df_player_stats["BIZ-Z"]
        # BIZ-I is intentionally left out
    )
    df_player_stats["PlaysMade"] = (
        df_player_stats["BIZ-Rm"]
        + df_player_stats["BIZ-Lm"]
        + df_player_stats["BIZ-Em"]
        + df_player_stats["BIZ-Um"]
        + df_player_stats["BIZ-Zm"]
    )
    df_player_stats["IPClean"] = ip_to_innings(df_player_stats["IP"])

    season_adj = _by_position(get_season_adjustment, pos)
    fielding_adj = _by_position(get_fielding_adjustment, pos)
    out_run_value = _by_position(get_out_run_value, pos)
    ip = df_player_stats["IPClean"]

    df_player_stats["PlaysMadeSeason"] = df_player_stats["PlaysMade"] / ip * season_adj

    df_lg_batting_stat, df_positional_totals = calculate_league_totals(df_player_stats)

    def lg(column: str) -> Series:
        return pos.map(df_positional_totals[column])

    df_player_stats["PlayMadeAASeason"] = df_player_stats["PlaysMadeSeason"] - lg("PlayPerSeason")
    df_player_stats["ARMAASeason"] = df_player_stats["ARM"] / ip * season_adj - lg("ARM")
    df_player_stats["FRMAASeason"] = df_player_stats["FRM"] / ip * season_adj - lg("FRM") * season_adj
    df_player_stats["EAASeason"] = df_player_stats["E"] / ip * season_adj - lg("ErrorPerSeason")
    df_player_stats["DPAASeason"] = df_player_stats["DP"] / ip * season_adj - lg("DP")

    df_player_stats["runsPSeason"] = (
        df_player_stats["PlayMadeAASeason"] * out_run_value
        - df_player_stats["EAASeason"] * out_run_value
        + df_player_stats["FRMAASeason"]
        + df_player_stats["ARMAASeason"] * out_run_value
    )

    df_player_stats["PlayMadeAANow"] = df_player_stats["PlaysMade"] - lg("PlayPerSeason") / season_adj * ip
    df_player_stats["ARMAANow"] = df_player_stats["ARM"] - lg("ARM") / season_adj * ip
    df_player_stats["FRMAANow"] = df_player_stats["FRM"] - lg("FRM") * ip
    df_player_stats["EAANow"] = df_player_stats["E"] - lg("ErrorPerSeason") / season_adj * ip
    df_player_stats["DPAANow"] = df_player_stats["DP"] - lg("DP") / season_adj * ip

    df_player_stats["runsPNow"] = (
        df_player_stats["PlayMadeAANow"] * out_run_value
        + df_player_stats["ARMAANow"] * out_run_value
        + df_player_stats["FRMAANow"]
        - (df_player_stats["EAANow"] * out_run_value)
    )

    df_player_stats["PlaysASeason"] = df_player_stats["PlaysAttempted"] / ip * season_adj
    df_player_stats["PlayAAASeason"] = df_player_stats["PlaysASeason"] - lg("PlayAttemptedPerSeason")
    df_player_stats["PlayPercent"] = df_player_stats["PlaysMade"] / df_player_stats["PlaysAttempted"]
    df_player_stats["PlayPercentAA"] = df_player_stats["PlayPercent"] - lg("PlayPercent")

    df_player_stats["wPMSeason"] = lg("PlayAttemptedPerSeason") * df_player_stats["PlayPercent"]
    df_player_stats["wPMAASeason"] = df_player_stats["wPMSeason"] - lg("PlayPerSeason")
    df_player_stats["wRunsPSeason"] = (
        df_player_stats["wPMAASeason"] * out_run_value
        - df_player_stats["EAASeason"] * out_run_value
        + df_player_stats["ARMAASeason"] * out_run_value
        + df_player_stats["FRMAASeason"]
    )
    df_player_stats["AdjustedPANow"] = lg("PlayAttemptedPerSeason") / fielding_adj * ip
    df_player_stats["wPMAdj"] = df_player_stats["AdjustedPANow"] * df_player_stats["PlayPercent"]
    df_player_stats["wPMAAAdj"] = df_player_stats["wPMAdj"] - (lg("PlayPerSeason") / fielding_adj * ip)
    df_player_stats["ARMAAAdj"] = df_player_stats["ARM"] - (lg("ARM") / fielding_adj * ip)
    df_player_stats["FRMAAAdj"] = df_player_stats["FRM"] - (lg("FRM") * ip)
    df_player_stats["EAAAdj"] = df_player_stats["E"] - lg("ErrorPerSeason") / season_adj * ip
    df_player_stats["DPAAdj"] = df_player_stats["DP"] - lg("DP") / fielding_adj * ip
    df_player_stats["runsPAdj"] = (
        df_player_stats["wPMAAAdj"] * out_run_value
        - df_player_stats["EAAAdj"] * out_run_value
        + df_player_stats["ARMAAAdj"] * out_run_value
        + df_player_stats["FRMAAAdj"]
    )

    df_player_stats["runsPAdjSeason"] = df_player_stats["wPMAAAdj"] / ip * fielding_adj

    df_fielding_attribute_averages = calculate_league_attribute_averages(df_player_stats)

    def lg_attr(column: str) -> Series:
        return pos.map(df_fielding_attribute_averages[column])

    df_player_stats["IFRngDelta"] = df_player_stats["IF RNG"] - lg_attr("Range")
    df_player_stats["IFArmDelta"] = df_player_stats["IF ARM"] - lg_attr("Arm")
    df_player_stats["IFErrDelta"] = df_player_stats["IF ERR"] - lg_attr("Error")
    df_player_stats["IFTDPDelta"] = df_player_stats["TDP"] - lg_attr("DP")
    df_player_stats["OFRngDelta"] = df_player_stats["OF RNG"] - lg_attr("Range")
    df_player_stats["OFArmDelta"] = df_player_stats["OF ARM"] - lg_attr("Arm")
    df_player_stats["OFERRDelta"] = df_player_stats["OF ERR"] - lg_attr("Error")
    df_player_stats["CFRMDelta"] = df_player_stats["C FRM"] - lg_attr("Frame")
    df_player_stats["CBLKDelta"] = df_player_stats["C ABI"] - lg_attr("Block")
    df_player_stats["CARMDelta"] = df_player_stats["C ARM"] - lg_attr("Arm")

    delta_columns = [
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
    ]
    df_player_stats[delta_columns] = df_player_stats[delta_columns].astype(float).round(2)

    df_player_stats = df_player_stats.dropna(subset=["runsPAdj", "runsPAdjSeason"])

    columns_to_remove = [x for x in df_player_ratings.columns if x not in ["ID", "POS", "Name"]]
    df_player_stats = df_player_stats.drop(columns=columns_to_remove)
    df_player_stats = df_player_stats.assign(**df_lg_stat["Value"].to_dict())

    return (
        df_positional_totals,
        df_fielding_attribute_averages,
        df_player_stats.set_index("ID"),
    )


def process_fielding_data(
    df_player_fielding_stats: DataFrame,
    df_lg_batting_stat: DataFrame,
    df_player_ratings: DataFrame,
    league: str,
    season: int,
) -> DataFrame:
    (
        df_league_fielding_totals,
        df_league_fielding_attributes,
        df_player_fielding_stats,
    ) = calculate_player_fielding_stats(df_player_fielding_stats, df_lg_batting_stat, df_player_ratings)
    df_player_fielding_stats["season"] = season
    df_player_fielding_stats["league"] = league
    # FRM is a per-inning rate, keep enough digits for it to survive rounding
    df_league_fielding_totals = df_league_fielding_totals.round(2).assign(FRM=df_league_fielding_totals["FRM"].round(6))
    df_league_fielding_totals.to_csv(paths.output_file(league, season, "fielding-totals"))
    df_league_fielding_attributes.to_csv(paths.output_file(league, season, "fielding-attributes"))
    df_player_fielding_stats.to_csv(paths.output_file(league, season, "fielding"))

    return df_player_fielding_stats
