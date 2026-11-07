import numpy as np
import pandas as pd
import pytest

from viewer import depth, injuries, player_card
from viewer.injuries import Injuries

SETTINGS = Injuries()


def test_the_time_left_is_read_by_the_upper_end_of_a_range_and_an_unknown_time_is_endless():
    assert injuries.days("1 day") == 1 and injuries.days("4 days") == 4 and injuries.days("one week") == 7
    assert injuries.days("2 weeks") == 14 and injuries.days("1-2 weeks") == 14 and injuries.days("2-3 weeks") == 21
    assert injuries.days("2-3 months") == 90 and injuries.days("14-15 months") == 450
    assert injuries.days("?") == np.inf and injuries.days("-") == np.inf and injuries.days(float("nan")) == np.inf


def test_day_to_day_out_and_the_injured_list_by_the_settings():
    states = {
        t: injuries.status(injuries.days(t), SETTINGS)
        for t in ("3 days", "one week", "8 days", "2 weeks", "2-3 weeks", "?")
    }
    assert states == {
        "3 days": "DTD",
        "one week": "DTD",
        "8 days": "OUT",
        "2 weeks": "OUT",
        "2-3 weeks": "IL",
        "?": "IL",
    }
    longer = Injuries(day_to_day=3, injured_list=30)
    assert (
        injuries.status(7, longer) == "OUT"
        and injuries.status(21, longer) == "OUT"
        and injuries.status(31, longer) == "IL"
    )


def test_the_settings_load_from_the_toml_and_reject_unknown_keys(tmp_path):
    assert injuries.load_injuries() == SETTINGS
    bad = tmp_path / "m.toml"
    bad.write_text("[injuries]\nnonsense = 1\n")
    with pytest.raises(ValueError):
        injuries.load_injuries(bad)
    good = tmp_path / "g.toml"
    good.write_text("[injuries]\ninjured_list = 30\n")
    assert injuries.load_injuries(good) == Injuries(day_to_day=7, injured_list=30)


def hurt_frame():
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 4],
            "INJ": ["-", "Yes", "Yes", "Yes", "Yes"],
            "Left": ["-", "4 days", "10 days", "3 months", "3 months"],
        }
    )


def test_the_injured_players_of_a_frame_get_a_state_and_a_text_and_the_well_ones_none():
    marks = injuries.marks(injuries.raw(hurt_frame()), SETTINGS)
    assert marks["status"].to_dict() == {2: "DTD", 3: "OUT", 4: "IL"} and marks["text"][2] == "DTD 4 days"
    assert marks["text"][4] == "IL 3 months" and injuries.listed(marks) == {4}
    assert injuries.raw(pd.DataFrame({"ID": [1]})).empty and injuries.listed(None) == set()


def test_a_table_gets_an_injury_column_after_the_name_only_when_somebody_in_it_is_hurt():
    marks = injuries.marks(injuries.raw(hurt_frame()), SETTINGS)
    table = pd.DataFrame({"ID": [1, 2, 3], "Player": list("abc"), "Age": 25})
    out = injuries.add(table, marks)
    assert list(out.columns) == ["ID", "Player", "Inj", "Age"] and out["Inj"].tolist() == [
        "",
        "DTD 4 days",
        "Out 10 days",
    ]
    assert injuries.add(table[table["ID"] == 1], marks) is not None and "Inj" not in injuries.add(
        table[table["ID"] == 1], marks
    )
    assert injuries.add(table, None) is table
    html = injuries.paint(out.style, out).to_html()
    assert injuries.COLORS["DTD"] in html and injuries.COLORS["OUT"] in html and injuries.COLORS["IL"] not in html
    plain = table.style
    assert injuries.paint(plain, table) is plain  # nothing to paint


def chart_frame():
    rows = []
    for player, slot, now, level, last in (
        (1, "LF", 5.0, "MLB", "a"),
        (2, "LF", 4.0, "MLB", "b"),
        (3, "LF", 3.0, "AAA", "c"),
    ):
        rows.append(
            {"ID": player, "slot": slot, "org": "ORG", "level_name": "MLB" if level == "MLB" else "AAA", "team_level": level,
             "First Name": "P", "Last Name": last, "Age": 27, "level": 1, "is_free_agent": False, "is_iafa": False,
             "is_draft": False, "T": 1, "B": 1, "value_now": now}
        )  # fmt: skip
    return pd.DataFrame(rows)


def test_the_depth_chart_leaves_out_the_injured_list_marks_the_others_and_lists_the_missing():
    frame = chart_frame()
    marks = injuries.marks(
        injuries.raw(pd.DataFrame({"ID": [1, 2], "INJ": ["Yes", "Yes"], "Left": ["3 months", "2 days"]})), SETTINGS
    )
    chart = depth.depth_chart(frame, "ORG", depth.WHOLE, marks=marks)
    assert chart["LF"]["ID"].tolist() == [2, 3] and chart["LF"]["Inj"].tolist() == ["DTD 2 days", ""]
    assert depth.depth_chart(frame, "ORG", depth.WHOLE)["LF"]["ID"].tolist() == [1, 2, 3]  # no marks: everybody
    listed = depth.injured_list(frame, "ORG", depth.WHOLE, marks)
    assert (
        listed["ID"].tolist() == [1] and listed["Inj"].tolist() == ["IL 3 months"] and listed["Lev"].tolist() == ["MLB"]
    )
    assert depth.injured_list(frame, "ORG", "AAA", marks).empty  # not at that level
    assert depth.injured_list(frame, "ORG", depth.WHOLE, None).empty


def test_the_planned_chart_marks_the_hurt_and_drops_a_listed_player_the_plan_still_has():
    frame = chart_frame()
    placed = pd.DataFrame(
        {"Level": "MLB", "Slot": ["LF", "BENCH"], "ID": [1, 2], "Player": ["P a", "P b"], "Primary": ["LF", "LF"],
         "total": 5.0, "Reason": "x"}
    )  # fmt: skip
    offered = pd.DataFrame({"Level": "MLB", "ID": [2], "slot": ["LF"], "total": [3.0]})
    marks = injuries.marks(
        injuries.raw(pd.DataFrame({"ID": [1, 2], "INJ": ["Yes", "Yes"], "Left": ["3 months", "2 days"]})), SETTINGS
    )
    chart = depth.planned_chart(frame, "ORG", "MLB", placed, offered, marks=marks)
    assert chart["LF"]["ID"].tolist() == [2] and chart["LF"]["Inj"].tolist() == ["DTD 2 days"]


def test_the_player_card_says_how_he_is_hurt_in_the_color_of_the_state():
    well = pd.Series({"INJ": "-", "Left": "-"})
    assert player_card.injury_html(well) == ""
    hurt = player_card.injury_html(pd.Series({"INJ": "Yes", "Left": "3 months"}))
    assert "IL" in hurt and injuries.COLORS["IL"] in hurt and "3 months" in hurt and "off the depth charts" in hurt
    assert injuries.COLORS["DTD"] in player_card.injury_html(pd.Series({"INJ": "Yes", "Left": "2 days"}))
