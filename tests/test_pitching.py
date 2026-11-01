import numpy as np
import pandas as pd
import pytest

from stats import pitching
from tests.helpers import (
    batting_league_totals,
    league_table,
    make_batting,
    make_pitching,
    make_ratings,
    pitching_league_totals,
)


def run_pitching(df, df_lg_woba=None):
    lg = league_table(pitching_league_totals(df))
    return (
        pitching.calculate_player_pitching_stats(df, lg, make_ratings(df["ID"]), "Test", df_lg_woba),
        lg,
    )


def test_role_classification():
    df = pd.DataFrame({"G": [10, 10, 10, 1], "GS": [10, 0, 4, 0]})
    assert pitching.classify_pitcher_role(df).tolist() == ["SP", "RP", "Mix", "RP"]


def test_usage_buckets_follow_the_share_of_starts():
    df = pd.DataFrame(
        {
            "G": [10, 10, 10, 10, 10, 10, 10, 1],
            "GS": [0, 1, 2, 3, 7, 8, 10, 0],
        }
    )
    # 0%, 10%, 20% | 30% | 70% | 80%, 100% | 0%
    assert pitching.classify_pitcher_usage(df).tolist() == [
        "RP",
        "RP-lean",
        "RP-lean",
        "Mix",
        "Mix",
        "SP-lean",
        "SP",
        "RP",
    ]


def test_usage_bucket_boundaries():
    df = pd.DataFrame({"G": [4, 4, 4, 4], "GS": [1, 2, 3, 4]})  # 25%, 50%, 75%, 100%
    assert pitching.classify_pitcher_usage(df).tolist() == ["RP-lean", "Mix", "Mix", "SP"]


def test_waa_sums_to_zero_within_each_baseline_bucket_and_overall():
    out, _ = run_pitching(make_pitching())
    assert set(out["PUsage"]) == {"SP", "RP", "Mix"}
    for bucket, group in out.groupby("PBaseline"):
        assert group["WAA"].sum() == pytest.approx(0, abs=1e-9), bucket
    assert out["WAA"].sum() == pytest.approx(0, abs=1e-9)


def test_a_starter_with_one_relief_appearance_is_not_judged_against_swingmen():
    df = make_pitching(n=60, seed=5)
    df["G"] = 30
    df["GS"] = [30, 29, 0, 15, 0, 28] * 10  # pure SP, SP-lean, RP, Mix, RP, SP-lean
    out, lg = run_pitching(df)
    assert set(out["PUsage"]) == {"SP", "SP-lean", "RP", "Mix"}
    by_usage = out.groupby("PUsage")["ExpRA9"].first()
    sp_lean = out[out["PUsage"] == "SP-lean"]
    assert (sp_lean["PBaseline"] == "SP-lean").all()
    assert sp_lean["ExpRA9"].nunique() == 1
    assert by_usage["SP-lean"] != by_usage["Mix"]


def test_small_buckets_merge_toward_mix():
    usage = pd.Series(["RP"] * 12 + ["RP-lean"] * 4 + ["Mix"] * 15 + ["SP-lean"] * 3 + ["SP"] * 11)
    baseline = pitching.resolve_baseline_buckets(usage)
    assert baseline.value_counts().to_dict() == {"Mix": 22, "RP": 12, "SP": 11}


def test_a_tiny_bucket_chain_merges_all_the_way_to_mix():
    usage = pd.Series(["RP"] * 4 + ["RP-lean"] * 5 + ["Mix"] * 20 + ["SP"] * 12)
    baseline = pitching.resolve_baseline_buckets(usage)
    assert baseline.value_counts().to_dict() == {"Mix": 29, "SP": 12}


def test_a_league_with_too_few_mixed_pitchers_uses_the_league_total():
    usage = pd.Series(["RP"] * 12 + ["Mix"] * 4 + ["SP"] * 12)
    baseline = pitching.resolve_baseline_buckets(usage)
    assert baseline.value_counts().to_dict() == {"RP": 12, "SP": 12, "Total": 4}


def test_league_totals_survive_a_missing_bucket():
    df = pd.DataFrame({"PBaseline": ["RP", "RP"], "IPClean": [20.0, 10.0], "R": [8, 3]})
    totals = pitching.calculate_league_totals(df)
    assert np.isnan(totals.loc["SP", "RA9"])
    assert totals.loc["RP", "RA9"] == pytest.approx(11 / 30 * 9)
    assert totals.loc["Total", "RA9"] == pytest.approx(11 / 30 * 9)


def test_role_column_is_unchanged_for_the_pitching_model():
    out, _ = run_pitching(make_pitching())
    assert set(out["PRole"]) == {"SP", "RP", "Mix"}


def test_rate_columns():
    df = make_pitching()
    out, _ = run_pitching(df)
    assert out["KPercent"].to_numpy() == pytest.approx((df["K"] / df["BF"] * 100).to_numpy())
    assert out["BBPercent"].to_numpy() == pytest.approx((df["BB"] / df["BF"] * 100).to_numpy())
    assert out["HRPercent"].to_numpy() == pytest.approx((df["HR"] / df["BF"] * 100).to_numpy())


def test_innings_use_baseball_notation():
    df = make_pitching()
    out, _ = run_pitching(df)
    whole = np.floor(df["IP"].to_numpy() + 1e-9)
    thirds = np.rint((df["IP"].to_numpy() - whole) * 10)
    assert out["IPClean"].to_numpy() == pytest.approx(whole + thirds / 3)


def test_woba_against_uses_the_batting_coefficients():
    df = make_pitching()
    batting_lg = league_table(batting_league_totals(make_batting(seed=9)))
    out_default, _ = run_pitching(df)
    out_batting, _ = run_pitching(df, batting_lg)
    assert not np.allclose(out_default["wOBAa"], out_batting["wOBAa"])
    c = batting_lg["Value"]
    den = df["AB"] + df["BB"] - df["IBB"] + df["SF"] + df["HP"]
    num = (
        df["SINGLE"] * c["coef_1B"]
        + df["DOUBLE"] * c["coef_2B"]
        + df["TRIPLE"] * c["coef_3B"]
        + df["HR"] * c["coef_HR"]
        + (df["BB"] - df["IBB"]) * c["coef_BB"]
        + df["HP"] * c["coef_HP"]
    )
    assert out_batting["wOBAa"].to_numpy() == pytest.approx((num / den).to_numpy())


def test_waaadj_is_alias_of_waa():
    out, _ = run_pitching(make_pitching())
    assert out["WAAAdj"].tolist() == out["WAA"].tolist()


def test_pitcher_with_runs_but_no_outs_is_charged_and_the_league_still_balances():
    df = make_pitching()
    df.loc[0, ["IP", "R", "K", "AB"]] = [0.0, 3, 0, 3]
    out, _ = run_pitching(df)
    row = out.loc[df.loc[0, "ID"]]
    assert row["IPClean"] == 0
    assert row["WAA"] == pytest.approx(-3 / 10)
    assert np.isnan(row["WAA200"])
    assert out["WAA"].sum() == pytest.approx(0, abs=1e-9)
