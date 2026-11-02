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


def test_waaadj_equals_waa_without_park_factors():
    out, _ = run_pitching(make_pitching())
    assert out["WAAAdj"].to_numpy() == pytest.approx(out["WAA"].to_numpy())


def test_pitcher_with_runs_but_no_outs_is_charged_and_the_league_still_balances():
    df = make_pitching()
    df.loc[0, ["IP", "R", "K", "AB"]] = [0.0, 3, 0, 3]
    out, _ = run_pitching(df)
    row = out.loc[df.loc[0, "ID"]]
    assert row["IPClean"] == 0
    assert row["WAA"] == pytest.approx(-3 / 10)
    assert np.isnan(row["WAA200"])
    assert out["WAA"].sum() == pytest.approx(0, abs=1e-9)


def test_fip_runs_sum_to_zero_per_bucket_and_follow_the_formula():
    out, _ = run_pitching(make_pitching())
    for bucket, group in out.groupby("PBaseline"):
        assert group["FIPRuns"].sum() == pytest.approx(0, abs=1e-9), bucket
        assert group["FIPRunsAdj"].sum() == pytest.approx(0, abs=1e-9), bucket
    numerator = 13 * out["HR"] + 3 * (out["BB"] + out["HP"] - out["IBB"]) - 2 * out["K"]
    assert out["FIPNumer"].to_numpy() == pytest.approx(numerator.to_numpy())
    for bucket, group in out.groupby("PBaseline"):
        rate = (13 * group["HR"] + 3 * (group["BB"] + group["HP"] - group["IBB"]) - 2 * group["K"]).sum()
        rate /= group["IPClean"].sum()
        expected = (rate * group["IPClean"] - group["FIPNumer"]) / 9
        assert group["FIPRuns"].to_numpy() == pytest.approx(expected.to_numpy())
    assert out["FIPWAA"].to_numpy() == pytest.approx((out["FIPRuns"] / 10).to_numpy())
    assert out["FIPRuns200"].to_numpy() == pytest.approx((out["FIPRuns"] * 200 / out["IPClean"]).to_numpy())


def test_pitcher_who_strikes_out_more_has_more_fip_runs():
    df = make_pitching()
    base, _ = run_pitching(df)
    better = df.copy()
    better.loc[3, "K"] += 25
    other, _ = run_pitching(better)
    pid = df.loc[3, "ID"]
    assert other.loc[pid, "FIPRuns"] > base.loc[pid, "FIPRuns"] + 4  # 25 K x 2 / 9 = 5.6 runs, less the league shift
    more_homers = df.copy()
    more_homers.loc[3, "HR"] += 5
    assert run_pitching(more_homers)[0].loc[pid, "FIPRuns"] < base.loc[pid, "FIPRuns"] - 6


def test_fip_runs_with_a_pitcher_without_outs():
    df = make_pitching()
    df.loc[0, ["IP", "R", "K", "AB"]] = [0.0, 3, 0, 3]
    out, _ = run_pitching(df)
    row = out.loc[df.loc[0, "ID"]]
    assert np.isfinite(row["FIPRuns"])
    assert np.isnan(row["FIPRuns200"])
    for _, group in out.groupby("PBaseline"):
        assert group["FIPRuns"].sum() == pytest.approx(0, abs=1e-9)


def test_fip_runs_adj_equals_fip_runs_without_park_factors_and_credits_hitter_parks():
    df = make_pitching()
    out, lg = run_pitching(df)
    assert out["FIPRunsAdj"].to_numpy() == pytest.approx(out["FIPRuns"].to_numpy())
    pid = df.loc[5, "ID"]
    parks = pd.DataFrame({"RUNS": 1.0}, index=pd.Index(df["ID"], name="ID"))
    parks.loc[pid, "RUNS"] = 1.2
    ratings = make_ratings(df["ID"])
    adjusted = pitching.calculate_player_pitching_stats(df, lg, ratings, "Test", None, parks)
    assert adjusted.loc[pid, "FIPRunsAdj"] > out.loc[pid, "FIPRuns"]  # a hitter park expects more runs
    for _, group in adjusted.groupby("PBaseline"):
        assert group["FIPRunsAdj"].sum() == pytest.approx(0, abs=1e-9)


def test_babip_runs_sum_to_zero_and_follow_the_formula():
    df = make_pitching()
    out, lg = run_pitching(df)
    assert out["BABIPRuns"].sum() == pytest.approx(0, abs=1e-9)
    hits = df["SINGLE"] + df["DOUBLE"] + df["TRIPLE"]
    bip = df["AB"] - df["K"] - df["HR"] + df["SF"]
    assert out["BIP"].to_numpy() == pytest.approx(bip.to_numpy())
    assert out["HitsBIP"].to_numpy() == pytest.approx(hits.to_numpy())
    c = lg["Value"]
    value = (
        df["SINGLE"].sum() * (c["run_value_1B"] + c["RUNS_MINUS"])
        + df["DOUBLE"].sum() * (c["run_value_2B"] + c["RUNS_MINUS"])
        + df["TRIPLE"].sum() * (c["run_value_3B"] + c["RUNS_MINUS"])
    ) / hits.sum()
    expected = (hits.sum() / bip.sum() - out["pBABIP"]) * out["BIP"] * value
    assert out["BABIPRuns"].to_numpy() == pytest.approx(expected.to_numpy())
    assert 0.6 < value < 1.0  # a hit is worth about 0.8 runs more than an out
    assert out["BABIPRuns200"].to_numpy() == pytest.approx((out["BABIPRuns"] * 200 / out["IPClean"]).to_numpy())


def test_babip_runs_react_to_hits_on_balls_in_play_not_to_home_runs():
    df = make_pitching()
    base, _ = run_pitching(df)
    pid = df.loc[7, "ID"]
    luckier = df.copy()
    luckier.loc[7, ["SINGLE", "HA"]] = [df.loc[7, "SINGLE"] - 10, df.loc[7, "HA"] - 10]
    assert run_pitching(luckier)[0].loc[pid, "BABIPRuns"] > base.loc[pid, "BABIPRuns"] + 6
    swap = df.copy()
    swap.loc[7, ["HR", "SINGLE"]] = [df.loc[7, "HR"] + 5, df.loc[7, "SINGLE"] - 5]  # same hits, 5 now leave the park
    unlucky = run_pitching(swap)[0].loc[pid]
    assert unlucky["BIP"] == base.loc[pid, "BIP"] - 5


def test_pitcher_with_no_balls_in_play_has_zero_babip_runs():
    df = make_pitching()
    df.loc[2, ["AB", "K", "HR", "SF", "SINGLE", "DOUBLE", "TRIPLE", "HA"]] = [4, 4, 0, 0, 0, 0, 0, 0]
    out, _ = run_pitching(df)
    assert out.loc[df.loc[2, "ID"], "BIP"] == 0
    assert out.loc[df.loc[2, "ID"], "BABIPRuns"] == 0
    assert out["BABIPRuns"].sum() == pytest.approx(0, abs=1e-9)


def test_waa_is_split_into_fip_babip_and_other_runs():
    out, _ = run_pitching(make_pitching())
    assert (out["FIPRuns"] + out["BABIPRuns"] + out["OtherRuns"]).to_numpy() == pytest.approx(
        (out["WAA"] * 10).to_numpy()
    )


COUNT_COLUMNS = [
    "K",
    "BF",
    "BB",
    "R",
    "ER",
    "SF",
    "SH",
    "HR",
    "IBB",
    "DP",
    "HA",
    "AB",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HP",
]


def split_in_two(df, share_right=0.7):
    """Right / left batter splits that add up to the total rows (IP is split in outs)."""
    right, left = df.copy(), df.copy()
    for column in COUNT_COLUMNS:
        right[column] = (df[column] * share_right).round().astype(int)
        left[column] = df[column] - right[column]
    outs = (df["IP"] // 1 * 3 + (df["IP"] % 1) * 10).round().astype(int)
    right_outs = (outs * share_right).round().astype(int)
    for frame, o in [(right, right_outs), (left, outs - right_outs)]:
        frame["IP"] = o // 3 + (o % 3) / 10
    return right, left


def test_pitching_split_values_sum_to_zero_per_side_and_bucket():
    df = make_pitching()
    total, lg = run_pitching(df)
    right, left = split_in_two(df)
    for side in (right, left):
        out = pitching.calculate_pitching_split(side, total, lg["Value"])
        for _, group in out.groupby("PBaseline"):
            assert group["FIPRuns"].sum() == pytest.approx(0, abs=1e-9)
        assert out["BABIPRuns"].sum() == pytest.approx(0, abs=1e-9)
        assert out["pBABIP"].dropna().between(0, 1).all()


def test_pitching_split_numerators_add_up_to_the_total():
    df = make_pitching()
    total, lg = run_pitching(df)
    right, left = split_in_two(df)
    out_right = pitching.calculate_pitching_split(right, total, lg["Value"])
    out_left = pitching.calculate_pitching_split(left, total, lg["Value"])
    assert (out_right["FIPNumer"] + out_left["FIPNumer"]).to_numpy() == pytest.approx(total["FIPNumer"].to_numpy())
    assert (out_right["BIP"] + out_left["BIP"]).to_numpy() == pytest.approx(total["BIP"].to_numpy())
    assert (out_right["HitsBIP"] + out_left["HitsBIP"]).to_numpy() == pytest.approx(total["HitsBIP"].to_numpy())


def test_pitching_split_uses_the_league_rate_of_its_own_side():
    df = make_pitching()
    total, lg = run_pitching(df)
    right, left = split_in_two(df)
    # a side that strikes out far more batters has a higher league rate, so the same strikeouts are worth less
    right["K"] = right["K"] * 2
    out = pitching.calculate_pitching_split(right, total, lg["Value"])
    plain = pitching.calculate_pitching_split(split_in_two(df)[0], total, lg["Value"])
    pid = df.loc[4, "ID"]
    bucket = total.loc[pid, "PBaseline"]
    peers = out[out["PBaseline"] == bucket]
    rate = peers["FIPNumer"].sum() / peers["IPClean"].sum()
    assert out.loc[pid, "FIPRuns"] == pytest.approx((rate * out.loc[pid, "IPClean"] - out.loc[pid, "FIPNumer"]) / 9)
    assert out.loc[pid, "FIPNumer"] < plain.loc[pid, "FIPNumer"]


def test_pitchers_without_a_total_row_are_left_out_of_the_split():
    df = make_pitching()
    total, lg = run_pitching(df)
    right, _ = split_in_two(df)
    extra = right.iloc[[0]].copy()
    extra["ID"] = 999999
    out = pitching.calculate_pitching_split(pd.concat([right, extra], ignore_index=True), total, lg["Value"])
    assert 999999 not in out.index
    assert len(out) == len(right)


def test_export_saves_without_pitcher_splits_write_nothing(tmp_path, monkeypatch):
    from util import paths

    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path)
    total, lg = run_pitching(make_pitching())
    right, left = pitching.process_pitching_splits(pd.DataFrame(), pd.DataFrame(), total, "T", 1, lg)
    assert right.empty and left.empty
    assert not any(tmp_path.rglob("*pitching*"))
