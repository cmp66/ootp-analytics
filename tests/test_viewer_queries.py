import numpy as np
import pandas as pd
import pytest

from model.outlook_scoring import exceed_probability
from scripts import outlook_lists
from viewer import queries


def test_exceed_probability_follows_the_quantiles_inside_and_outside_them():
    q = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]] * 5)  # quantiles at 10, 25, 50, 75, 90 %
    t = np.array([2.0, 0.0, 4.0, 4.5, -np.inf])
    p = exceed_probability(q, t)
    assert p[0] == pytest.approx(0.5) and p[1] == pytest.approx(0.9) and p[2] == pytest.approx(0.1)
    assert 0.1 > p[3] > 0.0  # beyond the 90th percentile, along the last segment (3.5 would be inside it: 0.175)
    assert p[4] == 1.0
    assert exceed_probability(q[:1], np.array([np.inf]))[0] == 0.0
    assert (
        exceed_probability(q[:1], np.array([-50.0]))[0] <= 1.0 and exceed_probability(q[:1], np.array([50.0]))[0] >= 0.0
    )


def make_frame():
    """Slot rows of six players: my organization (MYS) holds two shortstops and a starter; others are around."""
    rows = [
        # ID, org, level, slot, value_now, floor, q25, median, q75, ceiling, p10, p50, stay, depth, age, first, free agent
        (1, "MYS", 1, "SS", 10.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.01, 0.50, 1.0, False, 30),
        (2, "MYS", 1, "SS", 2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.00, 0.05, 1.0, False, 31),
        (3, "BOS", 3, "SS", -20.0, -25.0, -15.0, -5.0, 8.0, 20.0, 0.20, 0.40, 0.99, False, 20),
        (4, "BOS", 1, "SS", 15.0, 10.0, 13.0, 15.0, 17.0, 19.0, 0.40, 0.90, 1.0, False, 27),
        (5, "FA", 0, "SS", 12.0, 5.0, 10.0, 12.0, 14.0, 17.0, 0.30, 0.80, 0.9, False, 29),
        (6, "BOS", 6, "SS", -60.0, -70.0, -62.0, -55.0, -50.0, -40.0, 0.0, 0.0, 0.5, True, 17),
        (7, "MYS", 1, "SP", 6.0, 3.0, 5.0, 6.0, 7.0, 9.0, 0.02, 0.30, 1.0, False, 28),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["ID", "org", "level", "slot", "value_now", "floor_h3", "q25_h3", "value_h3", "q75_h3", "ceiling_h3",
                 "p_top10_h3", "p_top50_h3", "p_stay_h3", "depth_player", "Age"],
    )  # fmt: skip
    frame["rank_score_h3"] = 0.25 * frame["floor_h3"] + 0.5 * frame["value_h3"] + 0.25 * frame["ceiling_h3"]
    frame["level_name"] = frame["level"].map(lambda v: queries.LEVEL_NAMES.get(v, "other"))
    frame["is_free_agent"] = frame["org"] == "FA"
    frame["is_iafa"] = False
    frame["is_draft"] = False
    return frame


def test_pools_pick_my_organization_other_organizations_and_free_agents():
    frame = make_frame()
    assert set(queries.select_pool(frame, "org", "MYS")["ID"]) == {1, 2, 7}
    assert set(queries.select_pool(frame, "others", "MYS")["ID"]) == {3, 4, 6}  # not mine, not free agents
    assert set(queries.select_pool(frame, "free_agents")["ID"]) == {5}
    assert set(queries.select_pool(frame, "mlb")["ID"]) == {1, 2, 4, 7}
    with pytest.raises(ValueError):
        queries.select_pool(frame, "org")  # needs an organization
    with pytest.raises(ValueError):
        queries.select_pool(frame, "nope", "MYS")


def test_filters_by_slot_age_level_and_depth():
    frame = make_frame()
    assert set(queries.apply_filters(frame, slots=["SP"])["ID"]) == {7}
    assert set(queries.apply_filters(frame, age_min=28, age_max=30)["ID"]) == {1, 5, 7}
    assert set(queries.apply_filters(frame, levels=["AA", "Rookie"])["ID"]) == {3, 6}
    assert 6 not in set(queries.apply_filters(frame, include_depth=False)["ID"])


def test_lists_rank_by_chance_of_top_10_of_11_to_50_and_by_rank_score():
    frame = make_frame()
    very_good = queries.tier_list(frame, "p_top10", 3, top=3)
    assert very_good["ID"].tolist() == [4, 5, 3]
    good = queries.tier_list(frame, "p_11_50", 3, top=2)
    assert good["ID"].tolist() == [4, 5] and good["key"].round(2).tolist() == [0.5, 0.5]  # top 50 minus top 10
    assert (queries.tier_list(frame, "p_11_50", 3)["key"] >= 0).all()
    ranking = queries.tier_list(frame, "rank_score", 3, by_player=True)
    assert ranking["ID"].is_unique and ranking["ID"].iloc[0] == 4
    with pytest.raises(KeyError):
        queries.tier_list(frame, "p_unknown", 3)


def test_org_thresholds_use_the_kth_best_roster_player_of_each_slot():
    frame = make_frame()
    thresholds = queries.org_thresholds(frame, "MYS").set_index("slot")
    assert thresholds.loc["SS", "players"] == 2 and thresholds.loc["SS", "start_threshold"] == 10.0  # the SS to beat
    assert thresholds.loc["SS", "roster_threshold"] == 2.0  # the backup to beat
    assert thresholds.loc["SP", "start_threshold"] == -np.inf  # fewer than five starters: any starter beats nobody
    assert thresholds.loc["C", "start_threshold"] == -np.inf and thresholds.loc["C", "players"] == 0
    only_minors = frame.assign(level=3)
    assert (
        queries.org_thresholds(only_minors, "MYS").set_index("slot").loc["SS", "start_threshold"] == -np.inf
    )  # not on the 26


def test_fit_table_chances_follow_the_distribution_and_the_stay_chance():
    frame = make_frame()
    fit = queries.fit_table(frame, "MYS", 3).set_index("ID")
    assert fit.loc[4, "p_start_h3"] == pytest.approx(0.9)  # his floor (the 10th percentile) is the 10 he has to beat
    assert (
        fit.loc[4, "p_start_h3"] > fit.loc[5, "p_start_h3"] > 0.5
    )  # the free agent has a lower median and may not stay
    assert fit.loc[3, "p_start_h3"] < 0.45  # a prospect whose median is far below
    assert 0.0 < fit.loc[3, "p_start_h3"] < fit.loc[3, "p_roster_h3"] <= 0.99  # an easier bar for the backup place
    assert fit.loc[6, "p_start_h3"] < 0.01  # depth
    assert fit.loc[1, "p_start_h3"] == pytest.approx(0.5, abs=0.05)  # himself: about even against his own value now
    assert fit.loc[7, "p_start_h3"] == pytest.approx(1.0, abs=1e-6) and fit.loc[7, "start_threshold"] == -np.inf


def test_configured_lists_run_on_the_frame():
    defaults, specs = queries.load_list_specs()
    assert defaults["horizon"] == 5 and {"very_good", "good", "ranking", "draft", "iafa", "trade_targets"} <= set(specs)
    frame = make_frame()
    trade = queries.run_list(frame, specs["trade_targets"], 3, org="MYS", top=5)
    assert set(trade["ID"]) <= {3, 4} and trade["ID"].iloc[0] == 4  # not mine, not depth; he would start for my team
    assert queries.run_list(frame, specs["free_agents"], 3, top=5)["ID"].tolist() == [
        5
    ]  # the only free agent, aged 29 (the list wants 21 and older)
    with pytest.raises(ValueError):
        queries.run_list(frame, specs["trade_targets"], 3)  # needs an organization


def test_export_pool_flags_use_the_scouting_category_and_the_age_of_the_international_players():
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Age": [17, 24, 20, 25, 18],
            "level": [np.nan, np.nan, np.nan, 1.0, np.nan],
            "ORG": ["FA", "FA", "FA", "BOS", "FA"],
            "Lev": ["FA", "FA", "FA", "MLB", "FA"],
            "SctCat": ["International", "International", "Amateur", "Major", "Minor"],
        }
    )
    flags = queries.add_pool_flags(frame, "TGS27").set_index("ID")
    assert flags.loc[1, "is_iafa"] and not flags.loc[2, "is_iafa"]  # a 24-year-old foreign professional is a free agent
    assert flags.loc[3, "is_draft"] and not flags.loc[3, "is_free_agent"]
    assert flags.loc[2, "is_free_agent"] and flags.loc[5, "is_free_agent"] and not flags.loc[4, "is_free_agent"]
    assert flags.loc[4, "org"] == "BOS" and flags.loc[4, "level_name"] == "MLB"
    dump = queries.add_pool_flags(frame.assign(organization_id=[0, 0, 5, 7, 0]), "Model-Sim")
    assert (
        not dump["is_iafa"].any()
        and not dump["is_draft"].any()
        and dump["is_free_agent"].tolist() == [True, True, False, False, True]
    )


def test_the_list_script_prints_the_names_and_reports_bad_requests(capsys):
    assert outlook_lists.main(["TGS27", "--lists"]) == 0
    assert "very_good" in capsys.readouterr().out
    assert outlook_lists.main(["TGS27", "--list", "nope"]) == 2
    assert "unknown list" in capsys.readouterr().err
