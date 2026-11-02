import logging

import numpy as np
import pandas as pd
import pytest

from model import data
from util import paths

SAVE = "Fake"
SEASONS = [2046, 2047]


def write(season, kind, frame):
    path = paths.output_file(SAVE, season, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def output(tmp_path, monkeypatch):
    """A tiny league of output files: players 1 to 6 with known numbers in two seasons."""
    monkeypatch.setattr(paths, "FILES_ROOT", tmp_path)
    ids = [1, 2, 3, 4, 5, 6]
    for season in SEASONS:
        k = season - 2045
        ratings = pd.DataFrame(
            {
                "ID": ids,
                "B": [1, 2, 3, 1, 2, 1],
                "T": [1, 2, 1, 2, 1, 2],
                "BA vR": [50, 60, 55, 45, 40, 65],
                "BA vL": [48, 50, 57, 47, 42, 60],
                "GAP vR": 50,
                "GAP vL": 51,
                "POW vR": [40, 50, 60, 70, 55, 45],
                "POW vL": [41, 49, 58, 66, 50, 47],
                "EYE vR": 50,
                "EYE vL": 52,
                "K vR": 50,
                "K vL": 49,
                "SPE": [50, 60, 70, 55, 45, 40],
                "SR": 50,
                "STE": 50,
                "RUN": 50,
                "STU vR": [60, 55, 50, 45, 40, 62],
                "STU vL": [58, 56, 52, 44, 41, 60],
                "CON.1 vR": 50,
                "CON.1 vL": 50,
                "PBABIP vR": [50, 55, 45, 60, 52, 48],
                "PBABIP vL": [51, 54, 46, 59, 53, 47],
                "HRR vR": 50,
                "HRR vL": 50,
                "STM": 50,
                "IF RNG": [50, 60, 70, 40, 55, 65],
                "IF ARM": 50,
                "C FRM": [30, 40, 50, 60, 70, 20],
            }
        )
        write(season, "player-data", ratings)
        pa = np.array([600, 450, 120, 80, 300, 20]) * k
        write(
            season,
            "hitting",
            pd.DataFrame(
                {
                    "ID": ids,
                    "PA": pa,
                    "wRAA": [30, -9, 6, 4, -15, 1],
                    "wRAAAdj": [31, -8, 5, 3, -14, 2],
                    "BSR": [3, -1.5, 0.6, 0.4, 3, 0.1],
                }
            ),
        )
        for kind, column, wraa in [("right-hitting", "wRAARight", 20), ("left-hitting", "wRAALeft", 10)]:
            write(
                season,
                kind,
                pd.DataFrame({"ID": ids, "PA": [400, 300, 80, 50, 200, 10], column: [wraa, -6, 3, 2, -9, 1]}),
            )
        ip = np.array([180.0, 120.0, 60.0, 40.0, 25.0, 10.0])
        write(
            season,
            "pitching",
            pd.DataFrame(
                {
                    "ID": ids,
                    "IPClean": ip,
                    "G": [30, 40, 50, 60, 10, 5],
                    "GS": [30, 0, 0, 0, 10, 0],
                    "PBaseline": ["SP", "RP", "RP", "RP", "SP", "RP"],
                    "FIPRunsAdj": [12.0, -4.0, 3.0, 1.0, -2.0, 0.5],
                    "BIP": [500, 300, 150, 90, 60, 30],
                    "HitsBIP": [140, 100, 40, 27, 20, 9],
                }
            ),
        )
        for kind in ("right-pitching", "left-pitching"):
            write(
                season,
                kind,
                pd.DataFrame(
                    {
                        "ID": ids,
                        "IPClean": ip / 2,
                        "FIPRuns200": [5.0, -2.0, 1.0, 0.5, -1.0, 0.2],
                        "BIP": [250, 150, 75, 45, 30, 15],
                        "HitsBIP": [70, 50, 20, 13, 10, 5],
                    }
                ),
            )
        write(
            season,
            "fielding",
            pd.DataFrame(
                {
                    "ID": ids + ids,
                    "POS": [2] * 6 + [6] * 6,
                    "IPClean": [900, 800, 150, 300, 1000, 50] * 2,
                    "FldRunsSeason": list(np.arange(6) * 1.5) + list(np.arange(6) * -2.0),
                }
            ),
        )
    return tmp_path


def test_batting_target_is_adjusted_runs_per_600_pa_weighted_by_pa(output):
    table = data.batting_table(SAVE, SEASONS, ["BA vR", "BA vL"])
    first = table.frame[(table.frame["season"] == 2046) & (table.frame["ID"] == 1)].iloc[0]
    assert first["target"] == pytest.approx(31 / 600 * 600)
    assert first["weight"] == 600
    second = table.frame[(table.frame["season"] == 2047) & (table.frame["ID"] == 2)].iloc[0]
    assert second["target"] == pytest.approx(-8 / 900 * 600)
    assert table.features == ["BA vR", "BA vL"]
    assert table.X.shape[1] == 2 and len(table.y) == len(table.w) == len(table)


def test_minimum_volume_drops_small_samples(output):
    table = data.batting_table(SAVE, [2046], ["BA vR"], min_pa=100)
    assert sorted(table.frame["ID"]) == [1, 2, 3, 5]  # 80 and 20 PA are out
    assert len(data.batting_table(SAVE, [2046], ["BA vR"], min_pa=0)) == 6


def test_baserunning_target_uses_bsr(output):
    table = data.baserunning_table(SAVE, [2046], ["SPE", "SR", "STE", "RUN"], min_pa=100)
    row = table.frame[table.frame["ID"] == 5].iloc[0]
    assert row["target"] == pytest.approx(3 / 300 * 600)
    assert row["SPE"] == 45


def test_missing_ratings_are_dropped_and_logged_not_zero_filled(output, caplog):
    bad = pd.read_csv(paths.output_file(SAVE, 2046, "player-data"))
    bad["BA vR"] = bad["BA vR"].astype(object)
    bad.loc[bad["ID"] == 2, "BA vR"] = "-"
    bad.to_csv(paths.output_file(SAVE, 2046, "player-data"), index=False)
    with caplog.at_level(logging.WARNING):
        table = data.batting_table(SAVE, [2046], ["BA vR"], min_pa=100)
    assert 2 not in set(table.frame["ID"])
    assert "dropped 1 of 4 rows" in caplog.text


def test_unknown_rating_and_missing_file_are_clear_errors(output):
    with pytest.raises(KeyError, match="NOPE"):
        data.batting_table(SAVE, [2046], ["NOPE"])
    with pytest.raises(FileNotFoundError, match="process_seasons"):
        data.batting_table(SAVE, [2050], ["BA vR"])


def test_batting_split_rows_use_the_ratings_of_their_side(output):
    table = data.batting_split_table(SAVE, [2046], min_pa=30)
    frame = table.frame
    right = frame[(frame["side_left"] == 0) & (frame["ID"] == 4)].iloc[0]
    left = frame[(frame["side_left"] == 1) & (frame["ID"] == 4)].iloc[0]
    assert (right["BA"], left["BA"]) == (45, 47)  # vR then vL
    assert (right["POW"], left["POW"]) == (70, 66)
    assert right["target"] == pytest.approx(2 / 50 * 600)
    assert right["weight"] == 50
    assert set(frame["ID"]) == {1, 2, 3, 4, 5}  # player 6 has 10 PA against each side
    one = frame[frame["ID"] == 1]
    assert len(one) == 2 and set(one["side_left"]) == {0.0, 1.0}
    assert frame.groupby("ID")["bats_left"].nunique().max() == 1
    assert frame[frame["ID"] == 2]["bats_left"].eq(1).all()
    assert frame[frame["ID"] == 3]["bats_switch"].eq(1).all()
    assert table.features == ["BA", "GAP", "POW", "EYE", "K", "side_left", "bats_left", "bats_switch"]


def test_fip_table_target_is_runs_per_200_ip_and_roles_filter(output):
    table = data.pitching_fip_table(SAVE, [2046], ["STU vR", "T"], min_ip=30)
    row = table.frame[table.frame["ID"] == 1].iloc[0]
    assert row["target"] == pytest.approx(12.0 / 180 * 200)
    assert row["weight"] == 180
    assert row["start_share"] == 1.0
    assert sorted(table.frame["ID"]) == [1, 2, 3, 4]  # 25 and 10 IP are out
    starters = data.pitching_fip_table(SAVE, [2046], ["T"], min_ip=0, roles=("SP",))
    assert sorted(starters.frame["ID"]) == [1, 5]


def test_babip_target_is_centered_on_the_league_and_weighted_by_bip(output):
    table = data.pitching_babip_table(SAVE, [2046], ["PBABIP vR"], min_bip=100)
    league = 336 / 1130  # all hits on balls in play over all balls in play of the season file
    row = table.frame[table.frame["ID"] == 1].iloc[0]
    assert row["target"] == pytest.approx(140 / 500 - league)
    assert row["weight"] == 500
    assert sorted(table.frame["ID"]) == [1, 2, 3]


def test_pitching_split_rows_and_both_parts(output):
    fip = data.pitching_split_table(SAVE, [2046], "fip")
    frame = fip.frame
    right = frame[(frame["side_left"] == 0) & (frame["ID"] == 4)].iloc[0]
    left = frame[(frame["side_left"] == 1) & (frame["ID"] == 4)].iloc[0]
    assert (right["STU"], left["STU"]) == (45, 44)
    assert (right["PBABIP"], left["PBABIP"]) == (60, 59)
    assert right["throws_left"] == 1  # player 4 throws left (T == 2)
    assert right["target"] == pytest.approx(0.5)
    assert right["weight"] == 20
    assert 6 not in set(frame["ID"])  # 5 innings per side
    babip = data.pitching_split_table(SAVE, [2046], "babip")
    assert babip.frame["weight"].max() == 250
    league = (70 + 50 + 20 + 13 + 10 + 5) / (250 + 150 + 75 + 45 + 30 + 15)
    row = babip.frame[(babip.frame["ID"] == 1) & (babip.frame["side_left"] == 0)].iloc[0]
    assert row["target"] == pytest.approx(70 / 250 - league)
    with pytest.raises(ValueError):
        data.pitching_split_table(SAVE, [2046], "bogus")


def test_fielding_table_is_one_position_and_filters_innings(output):
    table = data.fielding_table(SAVE, [2046], 6, ["IF RNG", "IF ARM"], min_ip=200)
    assert sorted(table.frame["ID"]) == [1, 2, 4, 5]
    row = table.frame[table.frame["ID"] == 2].iloc[0]
    assert row["target"] == pytest.approx(-2.0)  # second shortstop row
    assert row["weight"] == 800
    assert table.meta["position"] == 6
    with pytest.raises(ValueError):
        data.fielding_table(SAVE, [2046], 1, ["IF RNG"])


def test_tables_stack_seasons_in_order(output):
    table = data.batting_table(SAVE, SEASONS, ["BA vR"], min_pa=0)
    assert table.frame.groupby("season").size().to_dict() == {2046: 6, 2047: 6}
    assert table.groups == "ID" and table.season == "season"


@pytest.fixture
def processed_fixture(tmp_path, monkeypatch):
    """The Model-Sim 2045 fixture season run through the real pipeline."""
    import shutil
    from pathlib import Path

    from stats import pipeline

    root = tmp_path / "files"
    shutil.copytree(Path(__file__).parent / "data", root)
    monkeypatch.setattr(paths, "FILES_ROOT", root)
    pipeline.process_season("Model-Sim", 2045)
    return root


@pytest.mark.integration
def test_every_table_builds_from_real_pipeline_output(processed_fixture):
    seasons = [2045]
    tables = [
        data.batting_table("Model-Sim", seasons, ["BA vR", "BA vL"], min_pa=50),
        data.baserunning_table("Model-Sim", seasons, ["SPE", "SR", "STE", "RUN"], min_pa=50),
        data.pitching_fip_table("Model-Sim", seasons, ["STU vR", "STU vL", "T"], min_ip=10),
        data.pitching_babip_table("Model-Sim", seasons, ["PBABIP vR"], min_bip=30),
        data.batting_split_table("Model-Sim", seasons, min_pa=10),
        data.pitching_split_table("Model-Sim", seasons, "fip"),
        data.pitching_split_table("Model-Sim", seasons, "babip"),
        data.fielding_table("Model-Sim", seasons, 6, ["IF RNG", "IF ARM"], min_ip=50),
        data.fielding_table("Model-Sim", seasons, 2, ["C FRM"], min_ip=50),
    ]
    for table in tables:
        assert len(table) > 20, table.name
        assert np.isfinite(table.frame[[*table.features, "target", "weight"]].to_numpy()).all(), table.name
        assert (table.w > 0).all()
    # a batter's split rows add up to his total: both sides' PA are in the total PA
    split = tables[4].frame
    total = data.batting_table("Model-Sim", seasons, ["BA vR"], min_pa=0).frame.set_index("ID")["weight"]
    by_player = split.groupby("ID")["weight"].sum()
    assert (by_player <= total.reindex(by_player.index) + 1e-9).all()
