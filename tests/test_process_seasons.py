import shutil
from pathlib import Path

import pytest

from scripts import process_seasons
from util import paths
from util.seasons import available_seasons, select_seasons

DATA_ROOT = Path(__file__).parent / "data"
SEASONS = [2045, 2046, 2047, 2048, 2049]


@pytest.fixture
def files_root(tmp_path, monkeypatch):
    root = tmp_path / "files"
    shutil.copytree(DATA_ROOT, root)
    monkeypatch.setattr(paths, "FILES_ROOT", root)
    return root


def test_default_is_the_latest_season():
    assert select_seasons(SEASONS) == [2049]


def test_a_single_season():
    assert select_seasons(SEASONS, season=2046) == [2046]


def test_start_and_stop_are_inclusive():
    assert select_seasons(SEASONS, start=2046, stop=2048) == [2046, 2047, 2048]
    assert select_seasons(SEASONS, start=2047, stop=2047) == [2047]


def test_start_alone_runs_through_the_latest_season():
    assert select_seasons(SEASONS, start=2047) == [2047, 2048, 2049]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stop": 2047}, "needs a start"),
        ({"start": 2048, "stop": 2046}, "after stop"),
        ({"season": 2044}, "not found"),
        ({"season": 2046, "start": 2045}, "either"),
        ({"start": 2060}, "no seasons from"),
        ({"start": 2040, "stop": 2046}, "without data"),
    ],
)
def test_bad_selections_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_seasons(SEASONS, **kwargs)


def test_a_gap_in_the_range_is_reported_not_skipped():
    with pytest.raises(ValueError, match=r"\[2047\]"):
        select_seasons([2045, 2046, 2048], start=2045, stop=2048)


def test_no_data_at_all_is_an_error():
    with pytest.raises(ValueError, match="no seasons found"):
        select_seasons([])


def test_available_seasons_for_a_dump_and_an_export_save(files_root):
    assert available_seasons("Model-Sim") == [2045]
    assert available_seasons("TGS27") == [2046]
    # a folder without a ratings subfolder (the output folder of another run, notes) is not a season
    (files_root / "TGS27" / "2047").mkdir()
    (files_root / "TGS27" / "notes").mkdir()
    assert available_seasons("TGS27") == [2046]
    (files_root / "TGS27" / "2047" / "ratings").mkdir()
    assert available_seasons("TGS27") == [2046, 2047]


@pytest.mark.integration
@pytest.mark.parametrize("save, season", [("Model-Sim", 2045), ("TGS27", 2046)])
def test_script_processes_the_latest_season_by_default(files_root, save, season, capsys):
    assert process_seasons.main([save]) == 0
    assert paths.output_file(save, season, "fielding").exists()
    assert paths.output_file(save, season, "pitching").exists()
    assert f"Processing {save} {season}" in capsys.readouterr().out


@pytest.mark.integration
def test_script_range_and_list(files_root, capsys):
    assert process_seasons.main(["TGS27", "--list"]) == 0
    assert "[2046]" in capsys.readouterr().out
    assert not paths.output_file("TGS27", 2046, "hitting").exists()  # listing writes nothing
    assert process_seasons.main(["TGS27", "--start", "2046", "--stop", "2046"]) == 0
    assert paths.output_file("TGS27", 2046, "hitting").exists()


def test_script_errors_return_a_nonzero_status(files_root, capsys):
    assert process_seasons.main(["Nope"]) == 2
    assert process_seasons.main(["TGS27", "--stop", "2046"]) == 2
    assert process_seasons.main(["TGS27", "--season", "1999"]) == 2
    assert "error:" in capsys.readouterr().err


def test_one_failing_season_does_not_stop_the_others(files_root, monkeypatch, capsys):
    (files_root / "TGS27" / "2047" / "ratings").mkdir(parents=True)
    calls = []

    def fake_process(save, season):
        calls.append(season)
        if season == 2046:
            raise RuntimeError("boom")

    monkeypatch.setattr(process_seasons, "process_season", fake_process)
    assert process_seasons.main(["TGS27", "--start", "2046"]) == 1
    assert calls == [2046, 2047]
    assert "Failed seasons: 2046" in capsys.readouterr().err
