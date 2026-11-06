import pandas as pd
import pytest

from util import ages, importCSV


def test_the_export_date_is_where_every_players_birthday_and_whole_age_agree():
    dob = pd.Series(["2000-03-10", "2000-03-11", "1999-01-01"])
    age = pd.Series([26, 25, 27])  # 26 on or after 2026-03-10, 25 before 2026-03-11, 27 anywhere in 2026
    first, last = ages.date_window(dob, age)
    assert first == pd.Timestamp("2026-03-10") and last == pd.Timestamp("2026-03-10")  # one day
    assert ages.export_date(dob, age) == pd.Timestamp("2026-03-10")
    loose = ages.date_window(pd.Series(["2000-03-10"]), pd.Series([26]))
    assert loose == (pd.Timestamp("2026-03-10"), pd.Timestamp("2027-03-09"))  # a single player leaves a year open
    assert ages.date_window(pd.Series([None, "x"]), pd.Series([20, 20])) is None
    first, last = ages.date_window(pd.Series(["2000-03-10", "2000-03-12"]), pd.Series([26, 26]))
    assert last >= first  # ages that fit one date
    bad = ages.date_window(pd.Series(["2000-03-10", "2000-03-08"]), pd.Series([26, 25]))
    assert bad[0] > bad[1]  # ages that do not fit one date give an empty window


def test_the_fine_age_has_the_game_age_as_its_whole_part_and_counts_the_share_of_the_year():
    dob = pd.Series(pd.to_datetime(["2000-03-10", "2000-03-10", "2000-02-29", "2000-03-10"]))
    on = pd.Series(pd.to_datetime(["2026-03-10", "2026-09-09", "2026-02-28", "2026-03-09"]))
    fine = ages.fine_age(dob, on)
    assert fine.iloc[0] == pytest.approx(26.0)  # on the birthday
    assert 26.49 < fine.iloc[1] < 26.51  # half a year later
    assert fine.iloc[2] == pytest.approx(26.0)  # 29 February counts as 28 February in a year without it
    assert 25.99 < fine.iloc[3] < 26.0  # the day before the birthday is still 25
    assert ages.fine_age(dob, "2026-03-10").iloc[0] == pytest.approx(26.0)  # one date for everybody


def test_game_dates_are_read_and_a_rating_is_never_dated_after_the_export():
    assert ages.parse_game_date("Feb. 26th   2046") == pd.Timestamp("2046-02-26")
    assert ages.parse_game_date("Oct. 1st 2081") == pd.Timestamp("2081-10-01")
    assert pd.isna(ages.parse_game_date("-")) and pd.isna(ages.parse_game_date(None))
    export = pd.Timestamp("2081-12-28")
    dates = ages.rating_dates(pd.Series(["Oct. 29th   2081", "Dec. 28th 2081", "?", "Jan. 3rd 2082"]), export)
    assert dates.tolist() == [pd.Timestamp("2081-10-29"), export, export, export]


def test_the_importer_adds_the_fine_age_at_the_date_of_each_players_ratings():
    frame = pd.DataFrame(
        {
            "DOB": ["03/10/2000", "03/11/2000", "01/01/1999"],
            "Age": [26, 25, 27],
            "Sct": ["Mar. 10th 2026", "Feb. 10th 2026", "Mar. 10th 2026"],
        }
    )
    out = importCSV.add_fine_age(frame, scouted=frame["Sct"])
    assert out["RatingDays"].tolist() == [0, 28, 0]  # the second was scouted four weeks before the export
    assert (
        out["AgeFine"].iloc[0] == pytest.approx(26.0) and 25.9 < out["AgeFine"].iloc[1] < 25.95
    )  # younger on his scouting date
    assert (out["AgeFine"].iloc[[0, 2]].astype(int) == [26, 27]).all()
    no_dob = importCSV.add_fine_age(pd.DataFrame({"Age": [20, 30]}))
    assert no_dob["AgeFine"].tolist() == [20.5, 30.5]  # without dates of birth: the age plus half a year
