import shutil
from pathlib import Path

import pandas as pd
import pytest

import stats
from stats import pipeline
from util import importCSV, paths

DATA_ROOT = Path(__file__).parent / "data"
SEASONS = [("Model-Sim", 2045), ("TGS27", 2046)]
OUTPUT_NAMES = [
    "player-data",
    "team-batting",
    "team-pitching",
    "team-fielding",
    "batting-totals",
    "hitting",
    "woba-calcs",
    "fielding-totals",
    "fielding-attributes",
    "fielding-tier-rates",
    "fielding",
    "pitching-totals",
    "pitching",
]


@pytest.fixture
def files_root(tmp_path, monkeypatch):
    """A scratch copy of the fixtures, so the pipeline's output files do not land in the repo."""
    root = tmp_path / "files"
    shutil.copytree(DATA_ROOT, root)
    monkeypatch.setattr(paths, "FILES_ROOT", root)
    return root


@pytest.mark.integration
@pytest.mark.parametrize("league, season", SEASONS)
def test_process_season_writes_every_output_once(files_root, league, season):
    result = pipeline.process_season(league, season)
    for name in OUTPUT_NAMES:
        assert paths.output_file(league, season, name).exists(), name
    written = pd.read_csv(paths.output_file(league, season, "player-data"), low_memory=False)
    assert "lgStuRating" in written.columns
    assert len(written) == len(result.player_data) == len(result.ratings)
    assert result.fielding["POS"].isin(range(1, 10)).all()


@pytest.mark.integration
@pytest.mark.parametrize("league, season", SEASONS)
def test_in_memory_player_data_matches_the_two_step_flow(files_root, league, season):
    # old flow: import_files writes player-data, add_league_attribute_averages reads it back
    result = pipeline.process_season(league, season)
    in_memory = pd.read_csv(paths.output_file(league, season, "player-data"), low_memory=False)

    importCSV.import_files(league, season)  # writes the plain ratings csv again
    stats.add_league_attribute_averages(league, season)
    two_step = pd.read_csv(paths.output_file(league, season, "player-data"), low_memory=False)

    pd.testing.assert_frame_equal(in_memory, two_step)
    assert result is not None


@pytest.mark.integration
@pytest.mark.parametrize("league, season", SEASONS)
def test_ability_columns_are_built_from_both_sources(files_root, league, season):
    result = pipeline.process_season(league, season)
    pitching, fielding = result.pitching, result.fielding
    for _, group in pitching.groupby("PBaseline"):
        assert group["FIPRuns"].sum() == pytest.approx(0, abs=1e-6)
        assert group["FIPRunsAdj"].sum() == pytest.approx(0, abs=1e-6)
    assert pitching["BABIPRuns"].sum() == pytest.approx(0, abs=1e-6)
    assert (pitching["FIPRuns"] + pitching["BABIPRuns"] + pitching["OtherRuns"]).to_numpy() == pytest.approx(
        (pitching["WAA"] * 10).to_numpy()
    )
    assert pitching["BIP"].ge(0).all()
    catchers = fielding[fielding["POS"] == 2]
    assert catchers["FldRunsSeason"].notna().all()
    # errors are reported apart from the fielding total
    assert (fielding["FldRunsAdj"] + fielding["ErrRunsAdj"]).to_numpy() == pytest.approx(
        fielding["runsPAdj"].to_numpy()
    )


@pytest.mark.integration
def test_dump_saves_write_pitcher_split_files_that_add_up_to_the_total(files_root):
    result = pipeline.process_season("Model-Sim", 2045)
    for name in ("right-pitching", "left-pitching"):
        assert paths.output_file("Model-Sim", 2045, name).exists()
    right, left, total = result.pitching_right, result.pitching_left, result.pitching
    for column in ["IPClean", "BF", "R", "K", "BB", "HR", "AB"]:
        assert right[column].sum() + left[column].sum() == pytest.approx(total[column].sum()), column
    for side in (right, left):
        for _, group in side.groupby("PBaseline"):
            assert group["FIPRuns"].sum() == pytest.approx(0, abs=1e-6)
        assert side["BABIPRuns"].sum() == pytest.approx(0, abs=1e-6)


@pytest.mark.integration
def test_export_saves_have_no_pitcher_split_files(files_root):
    result = pipeline.process_season("TGS27", 2046)
    assert result.pitching_right.empty and result.pitching_left.empty
    assert not paths.output_file("TGS27", 2046, "right-pitching").exists()
