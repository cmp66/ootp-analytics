from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from scripts.update_all import load_update_config
from util import paths
from util.seasons import available_seasons

APP = str(Path(__file__).resolve().parent.parent / "viewer" / "screens" / "explorer.py")
OVERVIEW = str(Path(__file__).resolve().parent.parent / "viewer" / "screens" / "org_overview.py")
ENTRY = str(Path(__file__).resolve().parent.parent / "viewer" / "app.py")


def has_outlook_files() -> bool:
    save = next(iter(load_update_config()))
    seasons = available_seasons(save)
    return bool(seasons) and paths.output_file(save, seasons[-1], "outlook-slots").exists()


pytestmark = pytest.mark.skipif(not has_outlook_files(), reason="needs the outlook files of scripts/update_all.py")


def test_app_shows_a_list_for_the_human_leagues():
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert len(app.dataframe[0].value) > 0 and len(app.dataframe) > 1  # the ordered list, then a table per position


def test_changing_the_list_changes_the_title():
    app = AppTest.from_file(APP, default_timeout=120).run()
    first = app.header[0].value
    app.sidebar.selectbox[2].select("draft").run()
    assert not app.exception and app.header[0].value != first


def test_a_player_appears_once_unless_positions_are_chosen_and_then_at_his_best_of_them():
    app = AppTest.from_file(APP, default_timeout=120).run()
    shown = app.dataframe[0].value
    named = shown.dropna(subset=["First Name", "Last Name"])  # a player called "Nan" or "NA" reads as missing
    assert not named.duplicated(["First Name", "Last Name", "Age"]).any()
    app.sidebar.multiselect[0].set_value(["SS"]).run()
    assert set(app.dataframe[0].value["slot"]) == {"SS"}


def test_the_report_has_a_block_per_position_and_prints_the_filters():
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Ordered list", "By position"]
    assert any("Positions" in m.value for m in app.markdown)


def test_report_functions_build_the_workbook():
    import io

    import openpyxl

    from viewer import queries, report

    save = next(iter(load_update_config()))
    season = available_seasons(save)[-1]
    frame = queries.load_outlook(save, season)
    pool = queries.apply_filters(queries.select_pool(frame, "mlb"), include_depth=False)
    spec = report.ReportSpec("Test", "rank_score", 3, [("Players from", "MLB players")], top=10, top_per_position=5)
    ordered, lowest, blocks = (
        report.ordered_list(pool, spec),
        report.ordered_list(pool, spec, True),
        report.position_blocks(pool, spec),
    )
    assert len(ordered) == 10 and ordered["key"].is_monotonic_decreasing and lowest["key"].is_monotonic_increasing
    assert "SS" in blocks and all(len(b) <= 5 for b in blocks.values())
    book = openpyxl.load_workbook(io.BytesIO(report.write_excel(ordered, lowest, blocks, spec)))
    assert book.sheetnames == ["Ordered list", "Lowest", "By position"]
    assert book["Ordered list"]["A2"].value == "Players from: MLB players"


def test_org_overview_fits_the_tables_asked_for():
    from viewer import overview, queries

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    org = load_update_config()[save]["org"]
    tables = overview.org_overview(frame, org)
    assert list(tables) == ["Position players", *overview.HITTER_SLOTS, "Starting pitchers", "Relief pitchers"]
    assert 0 < len(tables["SS"]) <= 10 and len(tables["Starting pitchers"]) > 5
    assert tables["Position players"]["Value"].is_monotonic_decreasing
    assert {"Off", "Def", "Run"} <= set(tables["Position players"].columns)
    assert not tables["Position players"]["Pos"].isin(["SP", "RP"]).any()
    assert set(tables["Relief pitchers"]["Lev"]) <= set(frame["level_name"])


def test_the_entry_point_opens_the_overview_for_my_org():
    app = AppTest.from_file(ENTRY, default_timeout=120).run()
    assert not app.exception
    org = load_update_config()[next(iter(load_update_config()))]["org"]
    assert (
        app.selectbox[0].value == next(iter(load_update_config())) and app.selectbox[1].value == org
    )  # league, then org


def test_the_long_tables_hold_the_whole_organization_and_carry_the_ids_for_the_ratings_card():
    from viewer import overview, queries

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    org = load_update_config()[save]["org"]
    tables = overview.org_overview(frame, org)
    pool = queries.select_pool(frame, "org", org)
    hitters = pool[~pool["slot"].isin(["SP", "RP"])]
    assert len(tables["Position players"]) == hitters["ID"].nunique() > 20
    assert {"ID", "Slot"} <= set(tables["Position players"].columns) and 0 < len(tables["SS"]) <= 10


def test_the_overview_lists_every_table_and_survives_a_rerun():
    app = AppTest.from_file(ENTRY, default_timeout=120).run()
    assert not app.exception and len(app.dataframe) == 12  # three long tables and nine positions
    app.selectbox[1].select(app.selectbox[1].options[1]).run()
    assert not app.exception


def test_the_ratings_page_has_the_sections_of_a_hitter_and_a_pitcher():
    from viewer import player_card

    save = next(iter(load_update_config()))
    row = pd.read_csv(paths.output_file(save, available_seasons(save)[-1], "player-data"), low_memory=False).set_index(
        "ID"
    )
    hitter = player_card.card_html(row.loc[13001], "LF")
    for text in (
        "Basic batting ratings",
        "Contact",
        "Gap Power",
        "Stealing Ability",
        "Left Field",
        "Outfield Range",
        "Personality",
    ):
        assert text in hitter
    assert "Catcher Ability" not in hitter and "Infield Range" not in hitter
    pitcher = player_card.card_html(row.loc[22544], "RP")
    for text in (
        "Basic pitching ratings",
        "Fastball",
        "HR Allowed",
        "PBABIP",
        "Arm Slot",
        "Velocity",
        "Pitcher Type",
        "Stamina",
    ):
        assert text in pitcher
    assert "Slider" not in pitcher


def test_rating_colors_and_bar_widths_follow_the_scale():
    from viewer import player_card

    assert player_card.color(40) == player_card.color(45) != player_card.color(50) != player_card.color(70)
    assert 'style="width:3.0%' in player_card.rating_row("x", 20) and "width:96.0%" in player_card.rating_row("x", 80)
    assert "/" in player_card.rating_row("x", 50, 60) and 'class="pot"' in player_card.rating_row("x", 50, 60)


def test_ratings_on_the_1_10_scale_get_the_bar_and_color_of_the_20_80_value_they_stand_for():
    from viewer import player_card

    assert player_card.as_20_80(1) == 20 and player_card.as_20_80(10) == 80 and player_card.as_20_80(0) == 0
    assert player_card.as_20_80(55) == 55 and player_card.as_20_80(5.5) == 50
    assert 'style="width:3.0%' in player_card.rating_row("x", 1) and "width:96.0%" in player_card.rating_row("x", 10)
    assert player_card.color(1) == player_card.color(20) and player_card.color(10) == player_card.color(80)
    assert player_card.color(5) == player_card.color(player_card.as_20_80(5))
    row = pd.Series({"POW": 8, "POW P": 10, "OVR": 6, "C": 0, "C Pot": 5, "SS": 7, "SS Pot": 9})
    page = player_card.card_html(row, "SS")
    assert ">8</span>" in page and ">10</span>" in page  # the numbers are shown as they are in the file
    catcher = pd.Series(
        {"C": 0, "C Pot": 5}
    )  # a potential of 5 on 1-10 is about 47 on 20-80: he can play there; 2 is about 27
    assert "Defensive ratings" in player_card.card_html(catcher, "C")
    assert "Defensive ratings" not in player_card.card_html(pd.Series({"C": 0, "C Pot": 2}), "C")


def test_each_screen_links_to_the_other():
    app = AppTest.from_file(ENTRY, default_timeout=120).run()
    assert not app.exception
    labels = [link.label for link in app.get("page_link")] if hasattr(app, "get") else []
    assert any("Explorer" in label for label in labels)


def test_minor_league_tables_leave_out_the_mlb_roster_and_use_floor_and_ceiling():
    from viewer import overview, queries

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    org = load_update_config()[save]["org"]
    ceiling = overview.org_overview(frame, org, "ceiling", minors_only=True)
    floor = overview.org_overview(frame, org, "floor", minors_only=True)
    for tables in (ceiling, floor):
        assert all("MLB" not in set(t["Lev"]) for t in tables.values())
    hitters = ceiling["Position players"]
    assert {"Ceiling", "Top-10 %"} <= set(hitters.columns) and "Value" not in hitters.columns
    assert hitters["Ceiling"].is_monotonic_decreasing and hitters["Top-10 %"].between(0, 100).all()
    assert {"Floor", "Top-50 %"} <= set(floor["Starting pitchers"].columns)
    everything = overview.add_projection(queries.select_pool(frame, "org", org))
    best = everything[everything["ID"] == hitters["ID"].iloc[0]]["peak_ceiling"].max()
    assert hitters["Ceiling"].iloc[0] == pytest.approx(best)


def test_the_minor_league_page_switches_between_ceiling_and_floor():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "viewer" / "screens" / "minors_overview.py"), default_timeout=120
    ).run()
    assert not app.exception and len(app.dataframe) == 12
    app.radio[0].set_value("Floor").run()
    assert not app.exception


def test_the_career_window_uses_the_career_columns_and_the_five_season_window_the_best_of_four():
    from viewer import overview

    pool = pd.DataFrame(
        {
            "ceiling_h1": [1.0], "ceiling_h3": [4.0], "ceiling_h5": [3.0], "ceiling_h99": [9.0],
            "floor_h1": [-5.0], "floor_h5": [-2.0], "floor_h99": [0.0],
            "p_top10_h5": [0.1], "p_top10_h99": [0.25], "p_top50_h5": [0.3], "p_top50_h99": [0.5],
        }
    )  # fmt: skip
    five, career = overview.add_projection(pool, "five"), overview.add_projection(pool, "career")
    assert five.loc[0, "peak_ceiling"] == 4.0 and five.loc[0, "best_top10"] == pytest.approx(10.0)
    assert career.loc[0, "peak_ceiling"] == 9.0 and career.loc[0, "peak_floor"] == 0.0
    assert career.loc[0, "best_top10"] == pytest.approx(25.0) and career.loc[0, "best_top50"] == pytest.approx(50.0)


def test_available_horizons_follow_the_columns_of_the_files():
    from viewer import queries

    old = pd.DataFrame(columns=[f"rank_score_h{h}" for h in (1, 2, 3, 5)])
    new = pd.DataFrame(columns=[f"rank_score_h{h}" for h in (1, 2, 3, 5, 7, 10, 99)])
    assert queries.available_horizons(old) == (1, 2, 3, 5) and queries.available_horizons(new)[-1] == queries.CAREER


SCREENS = Path(__file__).resolve().parent.parent / "viewer" / "screens"


@pytest.mark.parametrize("pool", ["draft", "iafa"])
def test_draft_and_iafa_tables_hold_only_their_pool_and_need_no_organization(pool):
    from viewer import overview, queries

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    tables = overview.pool_overview(frame, pool, None, "ceiling")
    ids = set(queries.select_pool(frame, pool)["ID"])
    assert tables["Position players"]["ID"].isin(ids).all() and set(tables) >= {
        "SS",
        "Starting pitchers",
        "Relief pitchers",
    }
    assert {"Ceiling", "Top-10 %"} <= set(tables["Position players"].columns)


@pytest.mark.parametrize("page", ["draft_overview", "iafa_overview", "minors_overview"])
def test_the_pool_pages_draw_twelve_tables_without_an_organization_selector_for_draft_and_iafa(page):
    app = AppTest.from_file(str(SCREENS / f"{page}.py"), default_timeout=120).run()
    assert not app.exception
    assert len(app.dataframe) == 12
    assert len(app.selectbox) == (
        2 if page == "minors_overview" else 1
    )  # the league, and the organization on the minors page


def test_all_organizations_lists_every_organization_with_an_org_column():
    from viewer import overview, queries

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    everyone = queries.select_pool(frame, "all_orgs")
    assert not everyone["org"].isin(["FA", "nan", ""]).any() and everyone["org"].nunique() > 20
    tables = overview.pool_overview(frame, "all_orgs", None, "current", with_org=True)
    assert "Org" in tables["Position players"].columns and tables["Position players"]["Org"].nunique() > 5
    assert "Org" in tables["SS"].columns and "Org" not in overview.org_overview(frame, "CLE")["SS"].columns
    minors = overview.pool_overview(frame, "all_orgs", None, "ceiling", minors_only=True, with_org=True)
    assert "MLB" not in set(minors["Position players"]["Lev"])


def test_free_agents_can_be_chosen_in_the_organization_dropdown():
    app = AppTest.from_file(str(SCREENS / "org_overview.py"), default_timeout=120).run()
    assert "Free agents" in app.selectbox[1].options
    app.selectbox[1].select("Free agents").run()
    assert not app.exception and len(app.dataframe) == 12
    assert "Org" not in app.dataframe[0].value.columns and len(app.dataframe[0].value) > 20


def test_the_position_menu_filters_the_position_players_table_but_not_the_other_tables():
    app = AppTest.from_file(str(SCREENS / "org_overview.py"), default_timeout=120).run()
    app.checkbox(key="pos-org-SS").check().run()
    assert not app.exception
    shown = app.dataframe[0].value
    assert set(shown["Pos"]) == {"SS"} and 0 < len(shown)
    assert len(app.dataframe) == 12 and len(app.dataframe[1].value) > 0  # the pitcher tables are untouched
    app.checkbox(key="pos-org-2B").check().run()
    assert set(app.dataframe[0].value["Pos"]) <= {"SS", "2B"} and len(app.dataframe[0].value) >= len(shown)


def test_the_position_filter_lists_every_player_who_qualifies_at_the_position_with_his_value_there():
    from viewer import overview, queries

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    org = load_update_config()[save]["org"]
    best_only = overview.org_overview(frame, org)["Position players"]
    at_short = overview.pool_overview(frame, "org", org, positions=["SS"])["Position players"]
    pool = queries.select_pool(frame, "org", org)
    qualified = pool[pool["slot"] == "SS"]
    assert (
        len(at_short) == len(qualified) > (best_only["Pos"] == "SS").sum()
    )  # not only the players whose best slot is SS
    assert at_short["Value"].is_monotonic_decreasing and set(at_short["Pos"]) == {"SS"}
    two = overview.pool_overview(frame, "org", org, positions=["SS", "2B"])["Position players"]
    assert set(two["Pos"]) <= {"SS", "2B"} and two["ID"].is_unique


def test_the_ratings_dialog_title_is_the_full_name():
    from viewer import player_card

    row = pd.Series({"First Name": "Jesus", "Last Name": "Recinos"})
    assert player_card.full_name(row, "J. Recinos") == "Jesus Recinos"
    assert player_card.full_name(pd.Series({"Age": 20}), "J. Recinos") == "J. Recinos"


DEPTH = str(Path(__file__).resolve().parent.parent / "viewer" / "screens" / "depth_chart.py")


def depth_frame():
    from viewer import queries

    save = next(iter(load_update_config()))
    return save, queries.load_outlook(save, available_seasons(save)[-1])


def test_depth_chart_lists_each_player_once_with_the_depth_asked_for():
    from viewer import depth

    save, frame = depth_frame()
    org = load_update_config()[save]["org"]
    chart = depth.depth_chart(frame, org)
    assert list(chart) == depth.CHART_SLOTS
    ids = [i for table in chart.values() for i in table["ID"]]
    assert len(ids) == len(set(ids))
    assert all(len(chart[s]) <= depth.DEPTH_HITTERS for s in depth.CHART_SLOTS if s not in ("SP", "RP"))
    assert all(len(chart[s]) <= depth.DEPTH_PITCHERS for s in ("SP", "RP"))
    assert 0 < len(chart["SP"]) <= depth.DEPTH_PITCHERS
    assert chart["SP"]["Value"].iloc[:5].is_monotonic_decreasing  # the first pass, best first
    assert chart["SP"]["Player"].str.match(r"^[RL]HP ").all()


def test_depth_chart_minor_leagues_leave_out_the_mlb_roster_and_rankings_change_the_order():
    from viewer import depth

    save, frame = depth_frame()
    org = load_update_config()[save]["org"]
    minors = depth.depth_chart(frame, org, "Minor leagues", "Ceiling (10 years)")
    assert all((table["Lev"] != "MLB").all() for table in minors.values())
    current = depth.depth_chart(frame, org, ranked_by="Current")
    floor = depth.depth_chart(frame, org, ranked_by="Floor (10 years)")
    assert not current["RP"]["ID"].equals(floor["RP"]["ID"]) or not current["C"]["ID"].equals(floor["C"]["ID"])
    with pytest.raises(ValueError):
        depth.depth_chart(frame, org, ranked_by="Best")


def test_depth_chart_page_draws_a_box_per_position_and_follows_the_choices():
    app = AppTest.from_file(DEPTH, default_timeout=120).run()
    assert not app.exception
    page = " ".join(str(e.value) for e in app.get("html"))
    for title in ("Left Field", "Shortstop", "Catcher", "Starting Rotation", "Bullpen"):
        assert title in page
    app.selectbox[2].select("Minor leagues").run()
    assert not app.exception
    app.selectbox[3].select("Floor (10 years)").run()
    assert not app.exception


def test_depth_assignment_puts_a_player_where_he_adds_the_most_and_only_where_he_is_ranked():
    import pandas as pd

    from viewer import depth

    # A is the better CF of the two but B is useless at RF, so A goes to CF; C is ranked only at LF
    values = pd.DataFrame({"CF": [10.0, 9.0, None], "RF": [8.0, 1.0, None], "LF": [None, None, 5.0]}, index=[1, 2, 3])
    assert sorted(depth.assign_pass(values, {"CF": 1, "RF": 1, "LF": 1})) == [(1, "RF"), (2, "CF"), (3, "LF")]
    # the best player of a position is not wasted elsewhere when the others cannot play it
    values = pd.DataFrame({"CF": [10.0, 9.0], "RF": [8.0, None]}, index=[1, 2])
    assert sorted(depth.assign_pass(values, {"CF": 1, "RF": 1})) == [(1, "RF"), (2, "CF")]
    # nobody is placed where he is not ranked, so a short position stays empty
    assert depth.assign_pass(pd.DataFrame({"CF": [None]}, index=[1]), {"CF": 1}) == []


def test_depth_chart_shows_one_level_at_a_time_and_flags_players_over_its_limits():
    from util.config import LevelLimits
    from viewer import depth

    save, frame = depth_frame()
    org = load_update_config()[save]["org"]
    options = depth.show_options(frame, org)
    assert options[:2] == list(depth.SHOW) and len(options) > 2
    level = options[2]
    chart = depth.depth_chart(frame, org, level, limits={})
    assert all((table["Lev"] == level).all() for table in chart.values())
    assert all((table["Flag"] == "").all() for table in chart.values())  # no limits: nobody is over
    strict = depth.depth_chart(frame, org, level, limits={level: LevelLimits(max_age=1, max_pro_years=1)})
    everybody = pd.concat(strict.values())
    assert len(everybody) and everybody["Flag"].str.contains("age").all()
    with pytest.raises(ValueError):
        depth.depth_chart(frame, org, "Nowhere")


def test_depth_chart_page_offers_the_levels_of_the_organization():
    app = AppTest.from_file(DEPTH, default_timeout=120).run()
    assert not app.exception
    shown = app.selectbox[2].options
    assert (
        any("International Complex" in option for option in shown) and len(shown) >= 3
    )  # more when the export has the minor leagues
    app.selectbox[2].select("AAA" if "Show: AAA" in shown else "INT").run()  # AAA when the export has the minor leagues
    assert not app.exception


def other_leagues_with_files():
    config = load_update_config()
    first = next(iter(config))
    found = []
    for name in list(config)[1:]:
        seasons = available_seasons(name)
        if seasons and paths.output_file(name, seasons[-1], "outlook-slots").exists():
            found.append(name)
    return first, found


@pytest.mark.parametrize(
    "page", ["org_overview", "depth_chart", "explorer", "minor_management", "acquisitions", "rule5"]
)
def test_the_league_selector_lists_every_save_and_switches_the_page_to_another_league(page):
    first, others = other_leagues_with_files()
    app = AppTest.from_file(str(SCREENS / f"{page}.py"), default_timeout=120).run()
    league_box = (app.sidebar.selectbox if page == "explorer" else app.selectbox)[0]
    assert list(league_box.options) == list(load_update_config())
    if not others:
        pytest.skip("only one league has outlook files")
    league_box.select(others[0]).run()
    assert not app.exception
    assert (app.sidebar.selectbox if page == "explorer" else app.selectbox)[0].value == others[0]


# ---------------------------------------------------------------- minor league management


def minors_inputs(changes: dict | None = None):
    """A small league: two organizations with ten hitters at each of MLB, AAA, AA and A; every key rating of a player is the same
    number (the level's typical 60 / 50 / 40 / 30, spread by 3 per place), ages 22, two years as a pro. `changes` sets columns of
    single players of MINE: {(level, place): {"Age": 26}}."""
    from util.config import LevelLimits

    rows, ratings, pid = [], [], 0
    for org in ("MINE", "OTHER"):
        for level, typical in (("MLB", 60), ("AAA", 50), ("AA", 40), ("A", 30)):
            for place in range(10):
                pid += 1
                value = typical + 3 * (place - 5)
                row = {"ID": pid, "season": 1, "slot": "SS", "value_now": float(value), "org": org, "team_level": level,
                       "level_name": level, "Age": 22, "PROY": 2, "First Name": "F", "Last Name": f"P{pid}",
                       "ceiling_h99": value + 8.0, "floor_h99": value - 8.0, "p_top50_h99": 0.1}  # fmt: skip
                if org == "MINE":
                    row.update((changes or {}).get((level, place), {}))
                rows.append(row)
                ratings.append({"ID": pid, **dict.fromkeys(("CON", "GAP", "POW", "EYE", "K's", "BABIP"), value)})
    levels = {"AAA": LevelLimits(), "AA": LevelLimits(), "A": LevelLimits(max_age=25)}
    return pd.DataFrame(rows), pd.DataFrame(ratings), levels


def moves(changes=None):
    from viewer import minors

    frame, ratings, levels = minors_inputs(changes)
    return minors.manage(frame, ratings, "MINE", levels)


def test_the_ladder_is_the_order_of_the_level_tables_without_the_levels_off_it():
    from util.config import LevelLimits
    from viewer import minors

    levels = {"AAA": LevelLimits(), "AA": LevelLimits(), "WL": LevelLimits(ladder=False), "R": LevelLimits()}
    assert minors.ladder(levels) == ["AAA", "AA", "R"]


def test_percentiles_count_ties_half_and_rules_come_from_the_config_file(tmp_path):
    import numpy as np

    from viewer import minors

    got = minors.percentile(pd.Series([1.0, 2.0, 3.0, 9.0]), np.array([1.0, 2.0, 2.0, 3.0]))
    assert got.round(3).tolist() == [0.125, 0.5, 0.875, 1.0]
    assert minors.load_rules(tmp_path / "missing.toml") == minors.Rules()
    config = tmp_path / "m.toml"
    config.write_text('[ratings]\nhitters = ["CON"]\n[rules]\ngreen = 0.8\n')
    rules = minors.load_rules(config)
    assert rules.green == 0.8 and rules.hitters == ("CON",) and rules.min_green_hitters == 4
    config.write_text("[rules]\nspeed = 1\n")
    with pytest.raises(ValueError):
        minors.load_rules(config)


def test_a_player_who_dominates_his_level_goes_up_and_a_weak_one_goes_down():
    from viewer import minors

    table = moves()
    assert set(table["Level"]) == {"AAA", "AA", "A"} and len(table) == 30  # MLB is not on the ladder
    # place 9 is the best of his level: green everywhere there and no worse than average at the level above
    best_aa = table[(table["Level"] == "AA") & (table["Value"] == 52.0)].iloc[0]
    assert best_aa["Move"] == minors.UP and best_aa["To"] == "AAA" and best_aa["green"] == 6
    worst_aa = table[(table["Level"] == "AA") & (table["Value"] == 25.0)].iloc[0]
    assert worst_aa["Move"] == minors.DOWN and worst_aa["To"] == "A" and worst_aa["Blocked"] == ""
    assert not ((table["Level"] == "A") & (table["Move"] == minors.DOWN)).any()  # nobody goes down from the bottom
    assert (table.loc[table["Move"] == "", ["To", "Reason", "Blocked"]] == "").all().all()  # no move, no reason


def test_a_player_over_the_limit_of_his_level_must_move_up_and_cannot_go_down_to_a_level_he_is_too_old_for():
    from viewer import minors

    table = moves({("A", 4): {"Age": 26}, ("AA", 0): {"Age": 26}})  # the A limit is 25; AA's worst player is 26
    old = table[(table["Level"] == "A") & (table["Age"] == 26)].iloc[0]
    assert old["Move"] == minors.MUST and old["To"] == "AA" and "age 26 > 25 at A" in old["Reason"]
    stuck = table[(table["Level"] == "AA") & (table["Age"] == 26)].iloc[0]
    assert stuck["Move"] == minors.DOWN and stuck["To"] == "A" and stuck["Blocked"] == "age 26 > 25 at A"


def test_the_management_page_lists_the_moves_of_every_league_and_survives_a_league_switch():
    app = AppTest.from_file(str(SCREENS / "minor_management.py"), default_timeout=180).run()
    assert not app.exception
    if len(app.dataframe) < 4:
        pytest.skip("the league has no minor leaguers on the ladder (an export without the minor league lists)")
    assert len(app.dataframe) >= 4
    assert any("to move" in s.value for s in app.subheader)
    assert {"Now", "After", "Max", "Pitchers after", "SP after"} <= set(
        app.dataframe[0].value.columns
    )  # the roster summary
    first, others = other_leagues_with_files()
    if others:
        app.selectbox[0].select(others[0]).run()
        assert not app.exception
    app.selectbox[1].select(app.selectbox[1].options[-1]).run()  # another organization
    assert not app.exception


def test_prospects_are_the_better_ceilings_with_development_left_and_the_rest_is_filler():
    from viewer import minors

    table = moves({("AA", 9): {"ceiling_h99": 300.0}, ("AA", 8): {"ceiling_h99": 300.0, "Age": 30}})
    top = table[(table["Level"] == "AA") & (table["Ceil"] == 308.0 - 8.0)] if False else table[table["Ceil"] == 300.0]
    assert sorted(top["Age"]) == [22, 30]
    young = top[top["Age"] == 22].iloc[0]
    assert (
        young["Tier"] == minors.TOP_PROSPECT and young["Window"] == 4 and bool(young["focus"])
    )  # hitters grow through 25
    old = top[top["Age"] == 30].iloc[0]
    assert (
        old["Tier"] == minors.PAST and old["Window"] == 0 and not old["focus"]
    )  # a great ceiling, but nothing left to develop
    assert set(table.loc[~table["focus"], "Tier"]) <= {minors.DEPTH, minors.PAST}
    assert 0.0 < table["focus"].mean() < 0.5  # the better fifth, not everybody


def test_the_development_age_is_set_per_game_version_and_kind_in_the_config():
    from viewer import minors

    rules = minors.load_rules()
    assert rules.develops_through("OOTP27", False) == 25 and rules.develops_through("OOTP26", True) == 26
    with pytest.raises(ValueError):
        rules.develops_through("OOTP99", False)


def test_the_acquisition_page_lists_the_needs_the_best_package_and_the_candidates_of_a_place():
    app = AppTest.from_file(str(SCREENS / "acquisitions.py"), default_timeout=240).run()
    assert not app.exception
    assert [s.value for s in app.subheader] == [
        "Priority list", "Where the roster needs an upgrade", "Best set of acquisitions within the budget space",
        "Candidates for a place",
    ]  # fmt: skip
    assert len(app.metric) == 4 and len(app.dataframe) >= 2
    priority = app.dataframe[0].value
    assert {"Impact", "Expected", "Verdict", "Spend", "Fallbacks"} <= set(priority.columns)
    assert set(priority["Spend"]) <= {"Within", "Over"}
    needs = next(t.value for t in app.dataframe if "Need" in t.value.columns)
    assert needs["Place"].iloc[0] and set(needs["Need"]) <= {"NEED", ""}
    app.checkbox[0].check().run()  # also upgrade places that are not a need
    assert not app.exception


@pytest.mark.parametrize(
    "page", ["org_overview", "minors_overview", "draft_overview", "depth_chart", "explorer", "acquisitions"]
)
def test_the_park_toggle_values_the_players_in_the_park_of_the_organization_and_follows_the_page(page):
    app = AppTest.from_file(str(SCREENS / f"{page}.py"), default_timeout=240).run()
    toggles = app.sidebar.toggle if page == "explorer" else app.toggle
    assert len(toggles) == 1 and toggles[0].label.startswith("Park-adjusted values")
    if toggles[0].disabled:
        pytest.skip("the league has no park factors or no park models (scripts/train_models.py --streams park)")
    assert not toggles[0].value
    toggles[0].set_value(True).run()
    assert not app.exception
    if page == "org_overview":
        assert "Park" in app.dataframe[0].value.columns  # what the park changes in each value
    if page == "acquisitions":
        assert any("Park" in t.value.columns for t in app.dataframe)  # the needs table


def test_the_park_toggle_is_off_for_all_organizations_and_free_agents():
    app = AppTest.from_file(str(SCREENS / "org_overview.py"), default_timeout=240).run()
    app.selectbox[1].select("Free agents").run()
    assert app.toggle[0].disabled and not app.exception


def test_an_exact_tie_between_assignments_is_decided_by_the_position_a_player_is_listed_at():
    from viewer import depth

    # A (listed SS) and B (listed 2B): SS + 2B for A + B and 2B + SS for A + B add up to the same -6.5
    values = pd.DataFrame({"SS": [8.0, -17.5], "2B": [11.0, -14.5]}, index=pd.Index([1, 2], name="ID"))
    listed = pd.Series({1: "SS", 2: "2B"})
    assert sorted(depth.assign_pass(values, {"SS": 1, "2B": 1}, listed)) == [(1, "SS"), (2, "2B")]
    assert sorted(depth.assign_pass(values, {"SS": 1, "2B": 1}, pd.Series({1: "2B", 2: "SS"}))) == [
        (1, "2B"),
        (2, "SS"),
    ]
    # a real difference in value still beats the preference
    better = pd.DataFrame({"SS": [8.0, -17.5], "2B": [12.0, -14.5]}, index=pd.Index([1, 2], name="ID"))
    assert sorted(depth.assign_pass(better, {"SS": 1, "2B": 1}, listed)) == [(1, "2B"), (2, "SS")]


def test_the_lists_carry_the_player_id_for_the_screens_but_not_for_the_excel_report():
    import io

    import openpyxl

    from viewer import queries, report

    save = next(iter(load_update_config()))
    frame = queries.load_outlook(save, available_seasons(save)[-1])
    pool = queries.apply_filters(queries.select_pool(frame, "mlb"), include_depth=False)
    spec = report.ReportSpec("Test", "rank_score", 3, [], top=5, top_per_position=3)
    plain = report.ordered_list(pool, spec)
    with_id = report.ordered_list(pool, spec, with_id=True)
    assert (
        "ID" not in plain.columns
        and with_id["ID"].is_unique
        and list(with_id.drop(columns="ID").columns) == list(plain.columns)
    )
    blocks = report.position_blocks(pool, spec, with_id=True)
    assert all("ID" in block.columns for block in blocks.values())
    book = openpyxl.load_workbook(io.BytesIO(report.write_excel(with_id, with_id, blocks, spec)))
    headers = {cell.value for row in book["Ordered list"].iter_rows(min_row=1, max_row=4) for cell in row}
    assert "ID" not in headers and "key" in headers


def test_the_explorer_and_the_acquisition_tables_keep_the_player_id_to_open_the_ratings_page():
    app = AppTest.from_file(str(SCREENS / "explorer.py"), default_timeout=240).run()
    assert not app.exception and "ID" in app.dataframe[0].value.columns
    app = AppTest.from_file(str(SCREENS / "acquisitions.py"), default_timeout=240).run()
    assert not app.exception and {"ID", "slot", "Place"} <= set(app.dataframe[0].value.columns)


def test_a_player_of_the_depth_chart_can_be_opened_from_the_list_below_the_filters():
    app = AppTest.from_file(DEPTH, default_timeout=240).run()
    box = next(s for s in app.selectbox if s.label == "Open a player's ratings")
    assert len(box.options) > 20 and box.options[0] == "Choose a player..."
    box.select(box.options[1]).run()
    assert not app.exception and app.session_state["depth_generation"] == 1  # the list is reset after the dialog opens


def test_the_depth_chart_of_the_mlb_roster_has_the_same_hitters_in_the_same_places_as_the_acquisitions_page():
    from viewer import acquire, depth

    save, frame = depth_frame()
    org = load_update_config()[save]["org"]
    options = depth.show_options(frame, org)
    assert "MLB" in options and depth.show_label("MLB") == "MLB roster" and "DH" in depth.CHART_SLOTS
    chart = depth.depth_chart(frame, org, "MLB")
    starters = acquire.needs(frame, org, acquire.load_settings())
    for slot in acquire.LINEUP:
        starter = starters[starters["place"] == slot]["ID"].iloc[0]
        top = chart[slot]["ID"].iloc[0] if len(chart[slot]) else float("nan")
        assert (pd.isna(starter) and pd.isna(top)) or starter == top, slot  # one lineup, two views


def test_the_acquisition_needs_say_where_each_starter_is_listed():
    from viewer import acquire

    save, frame = depth_frame()
    org = load_update_config()[save]["org"]
    needs = acquire.needs(frame, org, acquire.load_settings())
    assert "Listed" in needs.columns and needs.loc[needs["place"] == "SS", "Listed"].notna().all()


def platoon_frame():
    """Four hitters at two positions: the right-handed hitter is the better of the two against left-handed pitchers."""
    rows = []
    for player, slot, now, right, left, bats in (
        (1, "LF", 5.0, 8.0, -4.0, 2),  # lefty: better against right-handers
        (2, "LF", 4.0, 0.0, 12.0, 1),  # righty: better against left-handers
        (3, "RF", 3.0, 3.0, 3.0, 3),
        (4, "RF", 1.0, 1.0, 1.0, 3),
    ):
        rows.append(
            {
                "ID": player, "slot": slot, "org": "ORG", "level_name": "MLB", "team_level": "MLB", "First Name": "P",
                "Last Name": str(player), "Age": 27, "level": 1,
                "is_free_agent": False, "is_iafa": False, "is_draft": False, "T": 1, "B": bats,
                "value_now": now, "value_vs_right": right, "value_vs_left": left,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def test_the_lineup_against_each_side_picks_the_better_hitter():
    from viewer import depth

    frame = platoon_frame()
    assert depth.ranking_column(frame, "Current", "left") == "value_vs_left"
    with pytest.raises(ValueError):
        depth.ranking_column(frame, "Floor (10 years)", "left")  # the projections have no split
    with pytest.raises(ValueError):
        depth.ranking_column(frame.drop(columns="value_vs_left"), "Current", "left")
    right = depth.depth_chart(frame, "ORG", "MLB", side="right")
    left = depth.depth_chart(frame, "ORG", "MLB", side="left")
    assert right["LF"]["ID"].iloc[0] == 1 and left["LF"]["ID"].iloc[0] == 2
    assert right["LF"]["Value"].iloc[0] == 8.0 and left["LF"]["Value"].iloc[0] == 12.0
    assert depth.depth_chart(frame, "ORG", "MLB")["LF"]["ID"].iloc[0] == 1  # the overall lineup starts the lefty


def test_depth_chart_page_offers_the_overall_lineup_and_the_two_sides():
    app = AppTest.from_file(DEPTH, default_timeout=120).run()
    assert not app.exception
    lineup = next(box for box in app.selectbox if box.label == "Lineup")
    assert list(lineup.options) == ["Lineup: Overall", "Lineup: vs RHP", "Lineup: vs LHP"]
    for choice in ("vs RHP", "vs LHP"):
        lineup.select(choice).run()
        assert not app.exception
    app.selectbox[3].select("Floor (10 years)").run()  # the projections have no split: back to the overall chart
    assert not app.exception


def test_the_rule_5_page_lists_the_draft_eligible_players_the_roster_room_and_the_players_to_draft():
    app = AppTest.from_file(str(SCREENS / "rule5.py"), default_timeout=240).run()
    assert not app.exception
    assert [s.value for s in app.subheader] == [
        "Your players who can be drafted", "Who could come off the 40-man", "Players to draft from other organizations",
        "High-potential prospects to draft",
    ]  # fmt: skip
    assert len(app.metric) == 5 and len(app.dataframe) >= 2
    eligible = app.dataframe[0].value
    assert {"Player", "Action", "Why", "Tier"} <= set(eligible.columns) and set(eligible["Action"]) <= {
        "Protect",
        "Low risk",
        "Expose",
    }
    first, others = other_leagues_with_files()
    for (
        name
    ) in (
        others
    ):  # the other leagues: an exact 40-man where the export has the flag, an estimate (with a warning) where not
        app.selectbox[0].select(name).run()
        assert not app.exception
    app.number_input[0].set_value(3).run()
    assert not app.exception and app.metric[3].value >= "3"
