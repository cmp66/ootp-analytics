import numpy as np
import pandas as pd
import pytest

from util import importCSV


def write(path, df):
    df.to_csv(path, index=False)


def export_folder(tmp_path, with_team_batting=True):
    write(tmp_path / "pit.csv", pd.DataFrame({"ID": [1, 2], "IRS": [0, 1], "1B": [3, 4]}))
    write(
        tmp_path / "bat.csv",
        pd.DataFrame({"ID": [1, 2], "PI/PA": [3.9, 4.0], "K": [10, 12]}),
    )
    write(
        tmp_path / "field1.csv",
        pd.DataFrame({"ID": [1, 2], "POS": [2, 3], "BIZ-R": [1, 2]}),
    )
    write(
        tmp_path / "field2.csv",
        pd.DataFrame({"ID": [1, 3], "POS": [3, 4], "BIZ-R": [5, 6]}),
    )
    write(tmp_path / "tf.csv", pd.DataFrame({"ID": [1], "CERA": [4.0]}))
    write(tmp_path / "tp.csv", pd.DataFrame({"ID": [1], "WIN%": [0.5]}))
    if with_team_batting:
        write(tmp_path / "tb.csv", pd.DataFrame({"ID": [1], "wRC": [100]}))


def test_export_files_are_classified_by_their_marker_column(tmp_path):
    export_folder(tmp_path)
    tables = importCSV.classify_export_files(tmp_path)
    assert {name: len(frames) for name, frames in tables.items()} == {
        "team_fielding": 1,
        "team_pitching": 1,
        "team_batting": 1,
        "player_batting": 1,
        "player_pitching": 1,
        "player_fielding": 2,
    }


def test_missing_export_table_names_what_is_missing(tmp_path):
    export_folder(tmp_path, with_team_batting=False)
    with pytest.raises(FileNotFoundError, match="team_batting"):
        importCSV.classify_export_files(tmp_path)


def test_fielding_files_are_stacked_and_keys_checked(tmp_path):
    export_folder(tmp_path)
    tables = importCSV.classify_export_files(tmp_path)
    fielding = importCSV.combine_export_tables(tables["player_fielding"], ["ID", "POS"])
    assert len(fielding) == 4
    duplicated = [pd.DataFrame({"ID": [1, 1], "POS": [2, 2], "BIZ-R": [1, 2]})]
    with pytest.raises(ValueError, match="share the same"):
        importCSV.combine_export_tables(duplicated, ["ID", "POS"])


def test_exact_duplicate_rows_are_collapsed():
    frame = pd.DataFrame({"ID": [1, 1], "POS": [2, 2], "BIZ-R": [1, 1]})
    assert len(importCSV.combine_export_tables([frame], ["ID", "POS"])) == 1


def test_clean_ratings_zeroes_missing_ratings_but_keeps_text_columns():
    ratings = pd.DataFrame(
        {
            "CON": ["55", "-", "40"],
            "LF": ["-", "45", "50"],
            "COMP": ["Good", "-", "Fair"],
            "HSC": ["HS", "-", "CO"],
        },
        index=pd.Index([1, 2, 3], name="ID"),
    )
    cleaned = importCSV.clean_ratings(ratings)
    assert cleaned["CON"].tolist() == [55, 0, 40]
    assert cleaned["LF"].tolist() == [0, 45, 50]
    assert cleaned["CON"].dtype == np.int64
    assert cleaned["COMP"].tolist() == ["Good", "-", "Fair"]
    assert cleaned["HSC"].tolist() == ["HS", "-", "CO"]


def test_clean_ratings_never_zeroes_a_real_rating():
    ratings = pd.DataFrame({"POW": [20, 80, 45]}, index=pd.Index([1, 2, 3], name="ID"))
    assert importCSV.clean_ratings(ratings)["POW"].tolist() == [20, 80, 45]


def test_rename_tables_do_not_chain():
    # every source name must not also be a target name, otherwise renames would depend on order
    targets = set(importCSV.EXPORT_RATINGS_RENAMES.values())
    assert not targets & set(importCSV.EXPORT_RATINGS_RENAMES)


def test_stat_column_lists_have_no_duplicates():
    for columns in (
        importCSV.DUMP_BATTING_COLS,
        importCSV.DUMP_PITCHING_COLS,
        importCSV.DUMP_FIELDING_COLS,
    ):
        assert len(columns) == len(set(columns))


def test_team_innings_fold_extra_outs_into_ip():
    team = pd.DataFrame({"IP": [1000, 999], "IPF": [2, 0], "OTHER": [1, 2]})
    out = importCSV._team_innings(team)
    assert out["IP"].tolist() == pytest.approx([1000.2, 999.0])
    assert "IPF" not in out.columns


def test_missing_ratings_are_reported(caplog):
    stats = pd.DataFrame({"ID": [1, 2, 3], "PA": [100, 300, 100]})
    ratings = pd.DataFrame({"ORG": [1, 1]}, index=pd.Index([1, 2], name="ID"))
    with caplog.at_level("WARNING", logger="util.importCSV"):
        missing = importCSV.report_missing_ratings("batting", stats, ratings, "PA", "Test", 2000)
    assert missing == 1
    assert "1 of 3 players have no ratings" in caplog.text
    assert "20.00% of PA" in caplog.text


def test_nothing_is_logged_when_everyone_has_ratings(caplog):
    stats = pd.DataFrame({"ID": [1, 2], "PA": [100, 300]})
    ratings = pd.DataFrame({"ORG": [1, 1]}, index=pd.Index([1, 2], name="ID"))
    with caplog.at_level("WARNING", logger="util.importCSV"):
        assert importCSV.report_missing_ratings("batting", stats, ratings, "PA", "Test", 2000) == 0
    assert caplog.text == ""


def test_files_in_the_ratings_folder_that_are_not_player_lists_are_skipped(tmp_path, monkeypatch):
    import pandas as pd

    from util import importCSV, paths

    folder = tmp_path / "ratings"
    folder.mkdir()
    pd.DataFrame({"Team Name": ["A"], "Abbr": ["AAA"], "PF": [1.0]}).to_csv(folder / "park_info.csv", index=False)
    monkeypatch.setattr(paths, "export_ratings_dir", lambda league, season: folder)
    try:
        importCSV.import_ratings("L", 1, "R", str(tmp_path))
    except FileNotFoundError as error:
        assert "No ratings files" in str(error)  # nothing but the park file: no player list is left
    else:
        raise AssertionError("expected FileNotFoundError")
