import numpy as np
import pandas as pd
import pytest

from viewer import playing
from viewer.playing import Playing


def players():
    """Two shortstop prospects of one level (median age 21), a bench bat, and a pitcher."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Age": [19, 23, 27, 20],
            "pitcher": [False, False, False, True],
            "Window": [6, 2, 0, 6],
            "ceiling_pct": [0.95, 0.9, 0.1, 0.9],
            "slot": ["SS", "SS", "1B", "SP"],
            "d_cur": [np.nan, 2.0, np.nan, 2.0],
            "d_pot": [np.nan, 3.0, np.nan, 3.0],
            "std_SS": [0.8, 0.8, np.nan, np.nan],
            "std_2B": [0.7, 0.7, np.nan, np.nan],
            "std_1B": [np.nan, np.nan, 0.5, np.nan],
            "std_SP": [np.nan, np.nan, np.nan, 0.8],
            "rating_2B": [45.0, 45.0, np.nan, np.nan],
            "pot_2B": [60.0, 47.0, np.nan, np.nan],
            "rating_SS": [55.0, 68.0, np.nan, np.nan],  # player 1 is far from his SS potential, player 2 is close
            "pot_SS": [70.0, 70.0, np.nan, np.nan],
            "rating_1B": [np.nan, np.nan, 50.0, np.nan],
            "pot_1B": [np.nan, np.nan, 55.0, np.nan],
        }
    )


def test_the_settings_load_from_the_toml_with_the_measured_learning_rates_and_reject_unknown_keys(tmp_path):
    p = playing.load_playing()
    assert p.learn_by_level["rookie"] == pytest.approx(3.29) and p.learn_by_level["mlb"] == pytest.approx(0.08)
    assert p.carry_hitters == pytest.approx(0.80) and p.carry_pitchers < p.carry_hitters
    bad = tmp_path / "minors.toml"
    bad.write_text("[playing]\nnonsense = 1\n")
    with pytest.raises(ValueError):
        playing.load_playing(bad)
    partial = tmp_path / "p.toml"
    partial.write_text("[playing.learn_by_level]\nmlb = 0.5\n")
    loaded = playing.load_playing(partial)
    assert loaded.learn_by_level["mlb"] == 0.5 and loaded.learn_by_level["a"] == 1.51


def test_younger_for_the_level_counts_more_within_the_limits():
    p = Playing()
    factor = playing.age_factor([17, 21, 25, 40], 21, p)
    assert factor[1] == pytest.approx(1.0) and factor[0] > factor[1] > factor[2]
    assert factor[0] <= p.age_max and factor[3] == p.age_min


def test_growth_carries_over_by_the_measured_amounts_and_is_neutral_without_history():
    p = Playing()
    assert playing.expected_growth([1.0], [0.0], [False], p)[0] == pytest.approx(0.80)
    assert playing.expected_growth([1.0], [0.0], [True], p)[0] == pytest.approx(0.38)
    assert playing.expected_growth([np.nan], [np.nan], [False], p)[0] == 0.0
    assert (
        playing.momentum(np.array([0.0]), p)[0] == 1.0 and playing.momentum(np.array([100.0]), p)[0] == p.momentum_max
    )


def test_the_development_value_follows_ceiling_window_age_and_the_share_of_the_season():
    p = Playing()
    full = playing.development(0.9, 5, 21, 21, "SS", 0.0, p)
    assert full == pytest.approx(p.development * 0.9)
    assert playing.development(0.9, 5, 21, 21, "BENCH", 0.0, p) == pytest.approx(full * p.bench_share)
    assert playing.development(0.9, 0, 21, 21, "SS", 0.0, p) == 0.0  # no window left, no development value
    assert playing.development(0.9, 10, 21, 21, "SS", 0.0, p) == pytest.approx(full)  # the window counts to the cap
    assert playing.development(0.9, 5, 19, 21, "SS", 0.0, p) > full > playing.development(0.9, 5, 25, 21, "SS", 0.0, p)
    assert playing.development(0.9, 5, 21, 21, "SS", 5.0, p) == pytest.approx(full * 1.5)


def test_learning_is_slower_the_higher_the_level_and_stops_at_the_potential_at_the_position():
    p = Playing()
    rates = [playing.learning("2B", [30.0], [90.0], depth, p)[0] for depth in (0, 1, 3, 5)]
    assert rates == pytest.approx([0.8, 5.8, 15.1, 32.9]) and rates == sorted(rates)  # 1000 innings x the rate per 100
    assert playing.learning("2B", [45.0], [47.0], 5, p)[0] == pytest.approx(2.0)  # up to his potential only
    assert playing.learning("2B", [45.0], [np.nan], 5, p)[0] == 0.0  # no potential at the position: no learning
    assert playing.learning("2B", [np.nan], [60.0], 5, p)[0] == pytest.approx(32.9 if 60 - 25 > 32.9 else 60 - 25)
    assert (
        playing.learning("BENCH", [45.0], [60.0], 5, p)[0] == 0.0
        and playing.learning("DH", [45.0], [60.0], 5, p)[0] == 0.0
    )
    assert [playing.level_group(d) for d in (0, 1, 2, 3, 4, 5, 7)] == [
        "mlb",
        "upper",
        "upper",
        "a",
        "a",
        "rookie",
        "rookie",
    ]


def test_slot_utilities_list_only_the_slots_a_player_can_play():
    table = playing.slot_utilities(players(), median_age=21.0, depth=3)
    slots = table.groupby("ID")["slot"].apply(set)
    assert slots[1] == {"SS", "2B", "BENCH"} and slots[3] == {"1B", "BENCH"} and slots[4] == {"SP"}
    assert (table["total"] == table["development"] + table["learning"] - table["hit"] + table["current"]).all()


def test_learning_at_his_own_position_is_worth_more_and_most_at_the_key_positions():
    p = Playing()
    table = playing.slot_utilities(players(), median_age=21.0, depth=3).set_index(["ID", "slot"])
    home = table.loc[(1, "SS")]
    assert home["gain"] == pytest.approx(15.0) and home["primary"]
    assert home["learning"] == pytest.approx(p.learning * 15.0 * p.key_premium)  # SS is a key position
    first = table.loc[(3, "1B")]
    assert first["gain"] == pytest.approx(5.0) and first["learning"] == pytest.approx(
        p.learning * 5.0 * p.primary_premium
    )
    assert "trains his SS rating +15.0" in home["reason"]


def test_another_position_counts_in_full_only_once_the_primary_is_trained():
    p = Playing()
    table = playing.slot_utilities(players(), median_age=21.0, depth=3).set_index(["ID", "slot"])
    not_ready, ready = table.loc[(1, "2B")], table.loc[(2, "2B")]
    assert not not_ready["ready"] and ready["ready"]  # SS 55 of 70 (0.70) against 68 of 70 (0.96), key level 0.90
    assert not_ready["learning"] == pytest.approx(p.learning * not_ready["gain"] * p.before_ready)
    assert ready["learning"] == pytest.approx(p.learning * ready["gain"])
    assert "not trained enough yet" in not_ready["reason"] and "not trained enough yet" not in ready["reason"]


def test_an_unfamiliar_position_costs_performance_in_proportion_to_the_development_and_the_distance_to_the_potential():
    p = Playing()
    table = playing.slot_utilities(players(), median_age=21.0, depth=3).set_index(["ID", "slot"])
    off = table.loc[(1, "2B")]
    f = playing.familiarity([45.0], [60.0], p)[0]
    assert f == pytest.approx(25.0 / 40.0)
    assert off["hit"] == pytest.approx(p.unfamiliar_hit * off["development"] * (1 - f))
    assert table.loc[(1, "SS"), "hit"] == 0.0  # his own position costs nothing
    assert table.loc[(2, "2B"), "hit"] < off["hit"]  # the potential just above the rating: almost no hit
    assert playing.familiarity([np.nan], [np.nan], p)[0] == 0.0


def test_the_potential_caps_the_learning_in_a_table_and_the_bench_never_learns():
    table = playing.slot_utilities(players(), median_age=21.0, depth=3)
    capped = table[(table["ID"] == 2) & (table["slot"] == "2B")].iloc[0]
    assert capped["gain"] == pytest.approx(2.0)  # potential 47 against rating 45
    assert (table[table["slot"] == "BENCH"]["gain"] == 0).all()


def test_growth_and_age_lift_the_development_of_the_young_prospect_over_the_older_one():
    table = playing.slot_utilities(players(), median_age=21.0)
    ss = table[table["slot"] == "SS"].set_index("ID")
    assert ss.loc[1, "development"] > 0 and ss.loc[2, "development"] > 0
    # player 1 is 4 years younger with a longer window and the better ceiling; player 2 has growth (momentum) that player 1 lacks
    plain = playing.slot_utilities(players().assign(d_cur=np.nan, d_pot=np.nan), median_age=21.0)
    assert plain[(plain["ID"] == 2) & (plain["slot"] == "SS")]["development"].iloc[0] < ss.loc[2, "development"]
    assert plain[(plain["ID"] == 1) & (plain["slot"] == "SS")]["development"].iloc[0] == pytest.approx(
        ss.loc[1, "development"]
    )


def test_a_pitcher_has_only_pitching_slots_and_hitters_never_do():
    table = playing.slot_utilities(players(), median_age=21.0)
    assert set(table[table["ID"] == 4]["slot"]) == {"SP"}
    assert not table[table["slot"].isin(playing.PITCHER_SLOTS)]["ID"].isin([1, 2, 3]).any()
    assert playing.slot_utilities(players().iloc[0:0], median_age=21.0).empty
