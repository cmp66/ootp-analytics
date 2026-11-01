import pandas as pd
import pytest

from stats import batting, woba
from tests.helpers import (
    batting_league_totals,
    league_table,
    make_batting,
    make_ratings,
)


def run_batting(df_total, df_right=None, df_left=None):
    lg = league_table(batting_league_totals(df_total))
    ratings = make_ratings(df_total["ID"])
    empty = pd.DataFrame()
    total, right, left = batting.calculate_player_batting_stats(
        df_total,
        empty if df_right is None else df_right,
        empty if df_left is None else df_left,
        lg,
        ratings,
        "Test",
    )
    return total, right, left, lg


def test_woba_denominator_and_identity():
    df = make_batting()
    totals = batting_league_totals(df)
    lg = league_table(totals)["Value"]
    expected_den = df["AB"] + df["BB"] - df["IBB"] + df["SF"] + df["HP"]
    assert woba.woba_denominator(df).tolist() == expected_den.tolist()
    # the league wOBA is the same formula applied to the league totals
    assert lg["lgwOBA"] == pytest.approx(
        woba.woba_numerator(df, lg).sum() / expected_den.sum()
    )


def test_league_wraa_sums_to_zero():
    df = make_batting()
    total, _, _, _ = run_batting(df)
    assert total["wRAA"].sum() == pytest.approx(0, abs=1e-6)


def test_woba_coefficients_reproduce_league_run_scale():
    # one wOBA point above league is worth wOBA_SCALE-adjusted runs: wRAA of a player equals
    # (wOBA - lgwOBA) / scale * denominator
    df = make_batting()
    total, _, _, lg = run_batting(df)
    scale = lg.loc["wOBA_SCALE", "Value"]
    lg_woba = lg.loc["lgwOBA", "Value"]
    den = woba.woba_denominator(df).to_numpy()
    expected = (total["wOBA"].to_numpy() - lg_woba) / scale * den
    assert total["wRAA"].to_numpy() == pytest.approx(expected)


def test_obp_uses_standard_denominator():
    df = make_batting()
    total, _, _, _ = run_batting(df)
    expected = (df["H"] + df["BB"] + df["HP"]) / (
        df["AB"] + df["BB"] + df["HP"] + df["SF"]
    )
    assert total["OBP"].to_numpy() == pytest.approx(expected.to_numpy())


def test_split_offense_is_aligned_by_player_id():
    df = make_batting()
    # splits cover different players in a different order than the total
    right = df.sample(frac=0.8, random_state=1).reset_index(drop=True)
    left = df.iloc[::-1].iloc[5:].reset_index(drop=True)
    total, right_out, left_out, _ = run_batting(df, right, left)
    for split, suffix in [(right_out, "Right"), (left_out, "Left")]:
        expected = total["BSR"].reindex(split.index) + split[f"wRAA{suffix}"]
        assert split[f"OFF{suffix}"].to_numpy() == pytest.approx(expected.to_numpy())
        assert split.index.is_unique


def test_splits_use_the_same_wraa_formula():
    df = make_batting()
    total, right_out, _, _ = run_batting(df, df.copy(), df.copy())
    # a split containing everything the total contains must give the same wOBA / wRAA
    assert right_out["wOBARight"].to_numpy() == pytest.approx(total["wOBA"].to_numpy())
    assert right_out["wRAARight"].to_numpy() == pytest.approx(total["wRAA"].to_numpy())


def test_league_rates_do_not_double_count_ibb():
    df = make_batting()
    totals = batting_league_totals(df)
    lg = league_table(totals)["Value"]
    assert lg["lgK_RATE"] == pytest.approx(
        totals["SO"] / (totals["PA"] - totals["BB"] - totals["HP"])
    )
    assert lg["lgHR_RATE"] == pytest.approx(
        totals["HR"] / (totals["PA"] - totals["BB"] - totals["HP"])
    )


def test_ofadj_is_alias_of_off():
    total, _, _, _ = run_batting(make_batting())
    assert total["OFFAdj"].tolist() == total["OFF"].tolist()


def test_league_columns_are_added_once_per_stat():
    total, _, _, lg = run_batting(make_batting())
    assert not total.columns.duplicated().any()
    assert set(lg.index).issubset(total.columns)
