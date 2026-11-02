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
    _, _, out, _ = run_fielding(pd.concat([stats, dh], ignore_index=True), ratings)
    assert 1 in set(out["POS"])
    assert 10 not in set(out["POS"])


def test_missing_position_does_not_raise():
    stats, ratings = make_fielding(positions=[2, 3, 4, 5, 6, 7, 8, 9])  # nobody pitched
    _, _, out, _ = run_fielding(stats, ratings)
    assert 1 not in set(out["POS"])
    assert len(out) == len(stats)


def test_input_frame_is_not_modified():
    stats, ratings = make_fielding()
    before = stats.copy()
    run_fielding(stats, ratings)
    pd.testing.assert_frame_equal(stats, before)


def test_double_plays_do_not_change_fielding_runs():
    stats, ratings = make_fielding()
    _, _, base, _ = run_fielding(stats, ratings)
    changed = stats.assign(DP=stats["DP"] * 3 + 7)
    _, _, other, _ = run_fielding(changed, ratings)
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
    totals, _, out, _ = run_fielding(stats, ratings)
    catchers = out[out["POS"] == 2]
    # every catcher is exactly at the league rate, so framing runs above average are ~0
    assert catchers["FRMAAAdj"].abs().max() < 0.02
    assert catchers["FRMAASeason"].abs().max() < 0.02
    assert totals.loc[2, "FRM"] == pytest.approx(0.0005, rel=0.2)


def test_season_rate_columns_use_one_season_length():
    stats, ratings = make_fielding()
    _, _, out, _ = run_fielding(stats, ratings)
    catchers = out[out["POS"] == 2]
    expected = catchers["wPMAAAdj"] / catchers["IPClean"] * 1100
    assert catchers["runsPAdjSeason"].to_numpy() == pytest.approx(expected.to_numpy())


TIER_RATES = {"R": 0.95, "L": 0.75, "E": 0.5, "U": 0.25, "Z": 0.1}


def with_tier_mixes(mixes, position=4):
    """The league plus a position whose fielders all make every tier at the same rate but see different mixes."""
    stats, ratings = make_fielding()
    stats = stats[stats["POS"] != position]
    rows = []
    for i, mix in enumerate(mixes):
        row = {**stats.iloc[0].to_dict(), "ID": position * 1000 + i, "POS": position, "IP": 1000.0, "E": 5, "BIZ-I": 0}
        for tier, attempts in zip(TIER_RATES, mix):
            row[f"BIZ-{tier}"] = attempts
            row[f"BIZ-{tier}m"] = round(attempts * TIER_RATES[tier])
        rows.append(row)
    stats = pd.concat([stats, pd.DataFrame(rows)], ignore_index=True)
    ratings = ratings.reindex(stats["ID"].unique())
    ratings.iloc[:] = ratings.ffill().bfill()
    return stats, ratings


MIXES = [(400, 100, 60, 40, 20), (200, 160, 100, 80, 80), (600, 40, 20, 0, 0), (100, 100, 100, 100, 100)]


def test_tier_rates_match_an_independent_calculation():
    stats, ratings = make_fielding()
    rates = fielding.calculate_tier_rates(prepared(stats, ratings))
    for pos in range(1, 10):
        rows = stats[stats["POS"] == pos]
        for tier in "RLEUZ":
            attempts, made = rows[f"BIZ-{tier}"].sum(), rows[f"BIZ-{tier}m"].sum()
            assert rates.loc[pos, f"Att{tier}"] == attempts
            assert rates.loc[pos, f"Made{tier}"] == made
            assert rates.loc[pos, f"Rate{tier}"] == pytest.approx(made / attempts)


def test_sparse_tiers_use_the_all_position_rate():
    stats, ratings = make_fielding()
    stats.loc[stats["POS"] == 2, ["BIZ-Z", "BIZ-Zm"]] = [0, 0]  # nobody attempted tier Z at catcher
    stats.loc[stats["POS"] == 1, ["BIZ-U", "BIZ-Um"]] = [1, 0]  # 6 chances in the league, under the minimum
    rates = fielding.calculate_tier_rates(prepared(stats, ratings))
    for pos, tier in [(2, "Z"), (1, "U")]:
        pooled = stats[f"BIZ-{tier}m"].sum() / stats[f"BIZ-{tier}"].sum()
        assert rates.loc[pos, f"Rate{tier}"] == pytest.approx(pooled)
    assert not rates.filter(like="Rate").isna().any().any()


def test_expected_plays_follow_the_tier_rates():
    stats, ratings = make_fielding()
    df = prepared(stats, ratings)
    rates = fielding.calculate_tier_rates(df)
    expected = fielding.calculate_expected_plays(df, rates)
    for i in [0, 7, 20, 40]:
        row = df.iloc[i]
        by_hand = sum(row[f"BIZ-{t}"] * rates.loc[row["POS"], f"Rate{t}"] for t in "RLEUZ")
        assert expected.iloc[i] == pytest.approx(by_hand)


def test_plays_above_expected_sum_to_zero_at_every_position():
    stats, ratings = make_fielding()
    _, _, out, _ = run_fielding(stats, ratings)
    assert out.groupby("POS")["PlaysAboveExpected"].sum().abs().max() < 1e-6
    assert (out["PlaysExpected"] + out["PlaysAboveExpected"]).to_numpy() == pytest.approx(out["PlaysMade"].to_numpy())


def test_tier_mix_alone_does_not_change_the_fielders_value():
    stats, ratings = with_tier_mixes(MIXES)
    _, _, out, _ = run_fielding(stats, ratings)
    second_basemen = out[out["POS"] == 4]
    # same make rate in every tier: no matter the mix, nobody is above or below expected
    assert second_basemen["PlaysAboveExpected"].abs().max() < 1.0  # integer rounding of plays made
    assert second_basemen["PlayPercentAA"].abs().max() < 0.5
    assert second_basemen["runsPAdjSeason"].abs().max() < 5
    # while the overall make rate differs a lot between the mixes
    assert second_basemen["PlayPercent"].max() - second_basemen["PlayPercent"].min() > 0.15


def test_a_fielder_who_misses_chances_in_his_tier_is_below_expected():
    stats, ratings = with_tier_mixes(MIXES)
    victim = stats["ID"] == 4001
    stats.loc[victim, "BIZ-Lm"] -= 40  # misses 40 likely plays
    _, _, out, _ = run_fielding(stats, ratings)
    out = out[out["POS"] == 4]
    below = out.loc[4001]
    assert below["PlaysAboveExpected"] < -20  # the league rate he is measured against falls a little with his misses
    assert below["wPMAASeason"] < -20
    assert below["PlayPercentAA"] < -3
    assert out.drop(4001)["PlaysAboveExpected"].gt(0).all()


def test_tier_adjusted_columns_follow_their_definition():
    stats, ratings = make_fielding()
    totals, _, out, _ = run_fielding(stats, ratings)
    pos = out["POS"]
    lg = lambda column: pos.map(totals[column])  # noqa: E731
    exp_pct = out["PlaysExpected"] / out["PlaysAttempted"]
    assert out["PlayPercentExp"].to_numpy() == pytest.approx(exp_pct.to_numpy())
    assert out["PlayPercentAA"].to_numpy() == pytest.approx(((out["PlayPercent"] - exp_pct) * 100).to_numpy())
    expected = lg("PlayAttemptedPerSeason") * (out["PlayPercent"] - exp_pct)
    assert out["wPMAASeason"].to_numpy() == pytest.approx(expected.to_numpy())
    adj = lg("PlayAttemptedPerSeason") / pos.map(fielding.get_season_adjustment) * out["IPClean"]
    assert out["wPMAAAdj"].to_numpy() == pytest.approx((adj * (out["PlayPercent"] - exp_pct)).to_numpy())


def test_errors_in_the_make_rate_set_are_charged_per_chance():
    stats, ratings = make_fielding()
    # same errors per chance on twice the chances: the same error charge per season, but more errors in total
    first, second = stats.index[stats["POS"] == 4][:2]
    for column in [c for c in stats.columns if c.startswith("BIZ-")]:
        stats.loc[second, column] = stats.loc[first, column] * 2
    stats.loc[first, "E"], stats.loc[second, "E"] = 4, 8
    stats.loc[[first, second], "IP"] = 900.0
    _, _, out, _ = run_fielding(stats, ratings)
    one, two = out.loc[stats.loc[first, "ID"]], out.loc[stats.loc[second, "ID"]]
    assert two["EAAChanceSeason"] == pytest.approx(one["EAAChanceSeason"])
    # the volume based error column charges the extra errors
    assert two["EAASeason"] > one["EAASeason"] + 3


def test_volume_columns_keep_their_old_definition():
    stats, ratings = make_fielding()
    totals, _, out, _ = run_fielding(stats, ratings)
    pos = out["POS"]
    season = pos.map(fielding.get_season_adjustment)
    expected = out["PlaysMade"] / out["IPClean"] * season - pos.map(totals["PlayPerSeason"])
    assert out["PlayMadeAASeason"].to_numpy() == pytest.approx(expected.to_numpy())


def test_run_totals_of_the_make_rate_set_use_tier_plays_and_chance_errors():
    stats, ratings = make_fielding()
    _, _, out, _ = run_fielding(stats, ratings)
    value = out["POS"].map(fielding.get_out_run_value)
    adj = out["wPMAAAdj"] * value - out["EAAChanceAdj"] * value + out["ARMAAAdj"] * value + out["FRMAAAdj"]
    assert out["runsPAdj"].to_numpy() == pytest.approx(adj.to_numpy())
    season = (
        out["wPMAASeason"] * value - out["EAAChanceSeason"] * value + out["ARMAASeason"] * value + out["FRMAASeason"]
    )
    assert out["wRunsPSeason"].to_numpy() == pytest.approx(season.to_numpy())
    # the volume set keeps the old error term
    old = out["PlayMadeAANow"] * value + out["ARMAANow"] * value + out["FRMAANow"] - out["EAANow"] * value
    assert out["runsPNow"].to_numpy() == pytest.approx(old.to_numpy())
