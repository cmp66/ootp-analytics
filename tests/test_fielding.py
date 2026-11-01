import numpy as np
import pandas as pd
import pytest

from stats import fielding
from tests.helpers import (
    batting_league_totals,
    league_table,
    make_batting,
    make_fielding,
)


def run_fielding(stats, ratings):
    lg = league_table(batting_league_totals(make_batting()))
    return fielding.calculate_player_fielding_stats(stats, lg, ratings)


def prepared(stats, ratings):
    """Plays attempted / made and clean innings the way calculate_player_fielding_stats derives them."""
    df = stats.merge(ratings, on="ID")
    df["PlaysAttempted"] = df[["BIZ-R", "BIZ-L", "BIZ-E", "BIZ-U", "BIZ-Z"]].sum(axis=1)
    df["PlaysMade"] = df[["BIZ-Rm", "BIZ-Lm", "BIZ-Em", "BIZ-Um", "BIZ-Zm"]].sum(axis=1)
    df["IPClean"] = fielding.ip_to_innings(df["IP"])
    return df


def test_season_length_is_the_same_everywhere():
    assert [fielding.get_season_adjustment(p) for p in range(1, 10)] == [220, 1100] + [1200] * 7
    assert fielding.get_fielding_adjustment is fielding.get_season_adjustment


def test_positional_totals_match_an_independent_calculation():
    stats, ratings = make_fielding()
    df = prepared(stats, ratings)
    df["DP"] = df["DP"] * df["POS"].map(fielding.get_dp_adjustment)
    df["ARM"] = df["ARM"] * df["POS"].isin(fielding.of_positions)
    _, totals = fielding.calculate_league_totals(df)

    for pos in range(1, 10):
        rows = df[df["POS"] == pos]
        season = fielding.get_season_adjustment(pos)
        assert totals.loc[pos, "PlayPercent"] == pytest.approx(
            rows["PlaysMade"].sum() / rows["PlaysAttempted"].sum() * 100
        )
        assert totals.loc[pos, "PlayPerSeason"] == pytest.approx(
            rows["PlaysMade"].sum() / rows["IPClean"].sum() * season
        )
        assert totals.loc[pos, "ErrorPerSeason"] == pytest.approx(rows["E"].sum() / rows["IPClean"].sum() * season)
    catchers = df[df["POS"] == 2]
    assert totals.loc[2, "FRM"] == pytest.approx(catchers["FRM"].sum() / catchers["IPClean"].sum())
    assert totals.loc[2, "SBA"] == pytest.approx(catchers["SBA"].sum() / catchers["IPClean"].sum() * 1000)
    # only catchers carry catcher stats; the pitcher row is 0, not NaN
    assert totals.loc[[1, 3, 4, 5, 6, 7, 8, 9], ["SBA", "RTO", "FRM"]].eq(0).all().all()


def test_attribute_averages_are_innings_weighted():
    stats, ratings = make_fielding()
    df = prepared(stats, ratings)
    averages = fielding.calculate_league_attribute_averages(df)
    for pos in fielding.of_positions:
        rows = df[df["POS"] == pos]
        w = rows["IPClean"] / rows["IPClean"].sum()
        assert averages.loc[pos, "Range"] == pytest.approx(round((w * rows["OF RNG"]).sum(), 2))
        assert averages.loc[pos, "Arm"] == pytest.approx(round((w * rows["OF ARM"]).sum(), 2))
    catchers = df[df["POS"] == 2]
    w = catchers["IPClean"] / catchers["IPClean"].sum()
    assert averages.loc[2, "Frame"] == pytest.approx(round((w * catchers["C FRM"]).sum(), 2))


def test_pitchers_are_reported_and_dh_rows_are_not():
    stats, ratings = make_fielding()
    dh = stats[stats["POS"] == 3].copy()
    dh["POS"] = 10
    dh["ID"] = dh["ID"] + 5000
    ratings = pd.concat([ratings, ratings.loc[dh["ID"] - 5000].set_axis(dh["ID"].to_numpy())])
    ratings.index.name = "ID"
    _, _, out = run_fielding(pd.concat([stats, dh], ignore_index=True), ratings)
    assert 1 in set(out["POS"])
    assert 10 not in set(out["POS"])


def test_missing_position_does_not_raise():
    stats, ratings = make_fielding(positions=[2, 3, 4, 5, 6, 7, 8, 9])  # nobody pitched
    _, _, out = run_fielding(stats, ratings)
    assert 1 not in set(out["POS"])
    assert len(out) == len(stats)


def test_input_frame_is_not_modified():
    stats, ratings = make_fielding()
    before = stats.copy()
    run_fielding(stats, ratings)
    pd.testing.assert_frame_equal(stats, before)


def test_double_plays_do_not_change_fielding_runs():
    stats, ratings = make_fielding()
    _, _, base = run_fielding(stats, ratings)
    changed = stats.assign(DP=stats["DP"] * 3 + 7)
    _, _, other = run_fielding(changed, ratings)
    for column in [
        "runsPAdj",
        "runsPAdjSeason",
        "runsPSeason",
        "runsPNow",
        "wRunsPSeason",
    ]:
        assert other[column].to_numpy() == pytest.approx(base[column].to_numpy())
    # the reference columns still respond
    assert not np.allclose(other["DPAAdj"], base["DPAAdj"])


def test_catcher_framing_baseline_survives_rounding():
    stats, ratings = make_fielding()
    stats["FRM"] = np.where(stats["POS"] == 2, 0.0005 * stats["IP"], 0.0)  # tiny per-inning rate
    totals, _, out = run_fielding(stats, ratings)
    catchers = out[out["POS"] == 2]
    # every catcher is exactly at the league rate, so framing runs above average are ~0
    assert catchers["FRMAAAdj"].abs().max() < 0.02
    assert catchers["FRMAASeason"].abs().max() < 0.02
    assert totals.loc[2, "FRM"] == pytest.approx(0.0005, rel=0.2)


def test_season_rate_columns_use_one_season_length():
    stats, ratings = make_fielding()
    _, _, out = run_fielding(stats, ratings)
    catchers = out[out["POS"] == 2]
    expected = catchers["wPMAAAdj"] / catchers["IPClean"] * 1100
    assert catchers["runsPAdjSeason"].to_numpy() == pytest.approx(expected.to_numpy())
