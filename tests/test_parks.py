import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats import batting, parks, pitching
from tests.helpers import (
    batting_league_totals,
    league_table,
    make_batting,
    make_pitching,
    make_ratings,
    pitching_league_totals,
)
from util import paths
from util.config import SaveConfig

DATA_ROOT = Path(__file__).parent / "data"
COLUMNS = ["Team Name", "Park", "PF AVG", "AVG L", "AVG R", "PF HR", "HR L", "HR R", "PF D", "PF T", "PF"]


def park_csv(path, rows=None):
    rows = rows or [["Alpha", "A Park", 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.3, 1.4, 1.15], ["Beta", "B Park"] + [0.9] * 9]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


# ---- loading and validation -------------------------------------------------------------------------------------


def test_load_ballparks_renames_columns_and_indexes_by_team(tmp_path):
    loaded = parks.load_ballparks(park_csv(tmp_path / "p.csv"))
    assert loaded.index.tolist() == ["Alpha", "Beta"]
    assert loaded.loc["Alpha", ["AVG", "HR_L", "D", "T", "RUNS"]].tolist() == [1.1, 1.2, 1.3, 1.4, 1.15]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.drop(columns=["PF D"]), "missing columns"),
        (lambda df: pd.concat([df, df.iloc[[0]]]), "team listed twice"),
        (lambda df: df.assign(**{"PF HR": [None, 1.0]}), "empty park factor"),
        (lambda df: df.assign(**{"PF T": [9.0, 1.0]}), "outside"),
    ],
)
def test_bad_ballpark_files_are_rejected(tmp_path, mutate, message):
    path = tmp_path / "p.csv"
    mutate(pd.read_csv(park_csv(path))).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        parks.load_ballparks(path)


def test_leading_dots_in_the_real_file_format_parse(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(",".join(COLUMNS) + "\nAlpha,A Park,1.046,1.020,1.060,.838,.870,.820,1.315,1.040,.982\n")
    assert parks.load_ballparks(path).loc["Alpha", "HR"] == pytest.approx(0.838)


def test_ballpark_file_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path)
    (tmp_path / "S" / "2000").mkdir(parents=True)
    save = SaveConfig(name="S", source="dump", ratings="R", ballparks="ballpark_ratings.csv")
    assert parks.ballpark_file(SaveConfig(name="S", source="dump", ratings="R"), 2000) is None
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parks.ballpark_file(save, 2000)
    shared = park_csv(tmp_path / "S" / "ballpark_ratings.csv")
    assert parks.ballpark_file(save, 2000) == shared
    seasonal = park_csv(tmp_path / "S" / "2000" / "ballpark_ratings.csv")
    assert parks.ballpark_file(save, 2000) == seasonal


# ---- effective factors ------------------------------------------------------------------------------------------


def loaded(tmp_path):
    return parks.load_ballparks(park_csv(tmp_path / "p.csv"))


def test_effective_factor_blends_home_and_road(tmp_path):
    stints = pd.DataFrame({"ID": [1], "park": ["Alpha"], "weight": [100.0]})
    result = parks.effective_factors(stints, loaded(tmp_path), ["AVG_L", "RUNS"], home_share=0.5)
    assert result.loc[1, "AVG_L"] == pytest.approx(0.5 * 1.1 + 0.5)
    assert result.loc[1, "RUNS"] == pytest.approx(0.5 * 1.15 + 0.5)
    assert parks.effective_factors(stints, loaded(tmp_path), ["RUNS"], home_share=1.0).loc[1, "RUNS"] == pytest.approx(
        1.15
    )
    assert parks.effective_factors(stints, loaded(tmp_path), ["RUNS"], home_share=0.0).loc[1, "RUNS"] == pytest.approx(
        1.0
    )


def test_a_traded_player_is_weighted_by_playing_time(tmp_path):
    stints = pd.DataFrame({"ID": [1, 1], "park": ["Alpha", "Beta"], "weight": [300.0, 100.0]})
    result = parks.effective_factors(stints, loaded(tmp_path), ["RUNS"], home_share=1.0)
    assert result.loc[1, "RUNS"] == pytest.approx(0.75 * 1.15 + 0.25 * 0.9)


def test_unknown_park_and_zero_playing_time_are_neutral(tmp_path):
    stints = pd.DataFrame({"ID": [1, 2], "park": [np.nan, "Alpha"], "weight": [50.0, 0.0]})
    result = parks.effective_factors(stints, loaded(tmp_path), ["RUNS"], home_share=0.5)
    assert result["RUNS"].tolist() == [1.0, 1.0]


def test_export_stints_give_only_major_leaguers_a_park(tmp_path):
    ratings = pd.DataFrame(
        {"TM": ["AAA", "AAA", "BBB", "ZZZ"], "Lev": ["MLB", "AAA", "MLB", "MLB"]},
        index=pd.Index([1, 2, 3, 4], name="ID"),
    )
    stints = parks.export_stints(pd.Series([1, 2, 3, 4, 5]), ratings, loaded(tmp_path), {"AAA": "Alpha", "BBB": "Beta"})
    assert stints["park"].iloc[0] == "Alpha"
    assert stints["park"].iloc[2] == "Beta"
    assert (
        stints["park"].iloc[[1, 3, 4]].isna().all()
    )  # minor leaguer, unknown team abbreviation, player without ratings
    assert stints["expected"].tolist() == [True, False, True, True, False]


def test_unmatched_teams_are_logged(caplog):
    stints = pd.DataFrame(
        {
            "ID": [1, 2],
            "park": ["Alpha", np.nan],
            "team": ["a", "ZZZ"],
            "weight": [300.0, 100.0],
            "expected": [True, True],
        }
    )
    with caplog.at_level(logging.WARNING, logger="stats.parks"):
        parks.report_unmatched_parks(stints, "batting", "S", 2000)
    assert "ZZZ" in caplog.text
    assert "25.00%" in caplog.text


# ---- batting ----------------------------------------------------------------------------------------------------


def batting_with_parks(side=1, left=1.0, right=1.0, hr=1.0):
    """Run the batting calculation for a league where every batter bats from `side` and plays in one fixed park."""
    df = make_batting(n=40)
    lg = league_table(batting_league_totals(df))
    ratings = make_ratings(df["ID"], B=side)
    factors = pd.DataFrame(
        {"AVG_L": left, "AVG_R": right, "HR_L": hr, "HR_R": hr, "D": 1.0, "T": 1.0}, index=pd.Index(df["ID"], name="ID")
    )
    total, _, _ = batting.calculate_player_batting_stats(df, pd.DataFrame(), pd.DataFrame(), lg, ratings, "T", factors)
    return df, total


def test_neutral_factors_leave_offadj_equal_to_off():
    _, total = batting_with_parks()
    assert total["OFFAdj"].to_numpy() == pytest.approx(total["OFF"].to_numpy())
    assert total["wRAAAdj"].to_numpy() == pytest.approx(total["wRAA"].to_numpy())
    assert total[["ParkAVG", "ParkHR", "ParkD", "ParkT"]].eq(1.0).all().all()


def test_league_wraa_adj_sums_to_zero_with_uneven_parks():
    df = make_batting(n=40)
    lg = league_table(batting_league_totals(df))
    ids = df["ID"]
    rng = np.random.default_rng(0)
    factors = pd.DataFrame(
        {c: rng.uniform(0.85, 1.2, len(ids)) for c in parks.BATTING_FACTORS}, index=pd.Index(ids, name="ID")
    )
    total, _, _ = batting.calculate_player_batting_stats(
        df, pd.DataFrame(), pd.DataFrame(), lg, make_ratings(ids, B=1), "T", factors
    )
    assert total["wRAAAdj"].sum() == pytest.approx(0, abs=1e-6)


def test_hitter_friendly_park_lowers_offadj_and_pitcher_friendly_raises_it():
    df = make_batting(n=40)
    lg = league_table(batting_league_totals(df))
    half = np.arange(len(df)) % 2 == 0
    factors = pd.DataFrame(
        {
            "AVG_L": np.where(half, 1.15, 0.9),
            "AVG_R": np.where(half, 1.15, 0.9),
            "HR_L": 1.0,
            "HR_R": 1.0,
            "D": 1.0,
            "T": 1.0,
        },
        index=pd.Index(df["ID"], name="ID"),
    )
    total, _, _ = batting.calculate_player_batting_stats(
        df, pd.DataFrame(), pd.DataFrame(), lg, make_ratings(df["ID"], B=1), "T", factors
    )
    shift = total["OFFAdj"] - total["OFF"]
    assert shift[half].lt(0).all()
    assert shift[~half].gt(0).all()


def test_a_batter_uses_the_column_of_the_side_he_bats_from():
    _, as_lefty = batting_with_parks(side=2, left=1.3, right=1.0)
    _, as_righty = batting_with_parks(side=1, left=1.3, right=1.0)
    assert as_lefty["ParkAVG"].eq(1.3).all()
    assert as_righty["ParkAVG"].eq(1.0).all()


def test_switch_hitters_follow_the_pitchers_they_faced():
    df = make_batting(n=40)
    lg = league_table(batting_league_totals(df))
    ids = df["ID"]
    factors = pd.DataFrame(
        {"AVG_L": 1.4, "AVG_R": 1.0, "HR_L": 1.0, "HR_R": 1.0, "D": 1.0, "T": 1.0}, index=pd.Index(ids, name="ID")
    )
    right = df.copy()
    right["PA"] = 300  # plate appearances against right-handed pitchers (batting left)
    left = df.copy()
    left["PA"] = 100  # against lefties (batting right)
    total, _, _ = batting.calculate_player_batting_stats(df, right, left, lg, make_ratings(ids, B=3), "T", factors)
    assert total["ParkAVG"].to_numpy() == pytest.approx(0.75 * 1.4 + 0.25 * 1.0)
    # without split data a switch hitter is the average of both sides
    total, _, _ = batting.calculate_player_batting_stats(
        df, pd.DataFrame(), pd.DataFrame(), lg, make_ratings(ids, B=3), "T", factors
    )
    assert total["ParkAVG"].to_numpy() == pytest.approx(1.2)


def test_adjusted_singles_never_go_negative():
    df = make_batting(n=40)
    lg = league_table(batting_league_totals(df))
    ids = df["ID"]
    factors = pd.DataFrame(
        {"AVG_L": 0.5, "AVG_R": 0.5, "HR_L": 0.5, "HR_R": 0.5, "D": 0.4, "T": 0.4}, index=pd.Index(ids, name="ID")
    )
    total, _, _ = batting.calculate_player_batting_stats(
        df, pd.DataFrame(), pd.DataFrame(), lg, make_ratings(ids, B=1), "T", factors
    )
    assert np.isfinite(total["wOBAAdj"]).all()


# ---- pitching ---------------------------------------------------------------------------------------------------


def pitching_with_parks(factor):
    df = make_pitching()
    lg = league_table(pitching_league_totals(df))
    factors = pd.DataFrame({"RUNS": factor}, index=pd.Index(df["ID"], name="ID"))
    return df, pitching.calculate_player_pitching_stats(df, lg, make_ratings(df["ID"]), "T", None, factors)


def test_neutral_run_factor_leaves_waaadj_equal_to_waa():
    _, out = pitching_with_parks(1.0)
    assert out["WAAAdj"].to_numpy() == pytest.approx(out["WAA"].to_numpy())
    assert out["ParkRuns"].eq(1.0).all()


def test_waaadj_still_sums_to_zero_in_every_bucket_with_uneven_parks():
    df = make_pitching()
    factor = np.where(np.arange(len(df)) % 2 == 0, 1.12, 0.92)
    _, out = pitching_with_parks(factor)
    for bucket, group in out.groupby("PBaseline"):
        assert group["WAAAdj"].sum() == pytest.approx(0, abs=1e-9), bucket


def test_pitcher_friendly_park_lowers_waaadj_and_hitter_friendly_raises_it():
    df = make_pitching()
    even = np.arange(len(df)) % 2 == 0
    _, out = pitching_with_parks(np.where(even, 1.12, 0.92))
    shift = out["WAAAdj"] - out["WAA"]
    # a pitcher in a hitters' park is expected to allow more, so the same runs are worth more
    assert shift[even.tolist()].mean() > 0
    assert shift[(~even).tolist()].mean() < 0


# ---- whole season on the fixtures ------------------------------------------------------------------------------


@pytest.fixture
def files_root(tmp_path, monkeypatch):
    root = tmp_path / "files"
    shutil.copytree(DATA_ROOT, root)
    monkeypatch.setattr(paths, "FILES_ROOT", root)
    return root


@pytest.mark.integration
def test_neutral_save_gives_adj_equal_to_plain(files_root):
    from stats import pipeline

    result = pipeline.process_season("Model-Sim", 2045)
    # batters without plate appearances are NaN in both
    assert result.batting["OFFAdj"].to_numpy() == pytest.approx(result.batting["OFF"].to_numpy(), nan_ok=True)
    assert result.pitching["WAAAdj"].to_numpy() == pytest.approx(result.pitching["WAA"].to_numpy(), nan_ok=True)


@pytest.mark.integration
def test_real_parks_move_the_adj_stats_and_keep_the_league_balanced(files_root):
    from stats import pipeline

    result = pipeline.process_season("TGS27", 2046)
    batters, pitchers = result.batting, result.pitching
    assert (batters["OFFAdj"] - batters["OFF"]).abs().max() > 1
    assert batters["wRAAAdj"].sum() == pytest.approx(0, abs=1e-6)
    assert (pitchers["WAAAdj"] - pitchers["WAA"]).abs().max() > 0.1
    for bucket, group in pitchers.groupby("PBaseline"):
        assert group["WAAAdj"].sum() == pytest.approx(0, abs=1e-6), bucket
    # hitters in the most hitter-friendly park lose runs, hitters in the most pitcher-friendly park gain
    q = batters[batters["PA"] >= 200]
    assert q.loc[q["ParkAVG"].idxmax(), "OFFAdj"] < q.loc[q["ParkAVG"].idxmax(), "OFF"]
    assert q.loc[q["ParkAVG"].idxmin(), "OFFAdj"] > q.loc[q["ParkAVG"].idxmin(), "OFF"]


@pytest.mark.integration
def test_dump_teams_match_the_ballpark_names(files_root):
    table = parks.load_ballparks(paths.save_dir("Model-Sim") / "ballpark_ratings.csv")
    team_parks, _ = parks.dump_team_parks("Model-Sim", 2045, table)
    assert team_parks.notna().sum() == len(table) == 30
    assert team_parks.dropna().is_unique
