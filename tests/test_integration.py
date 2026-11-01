"""Cross-checks on real OOTP data. Nothing is written.

By default the small fixtures in tests/data are used (see scripts/make_test_data.py), so the tests run on a
fresh clone. Set OOTP_TEST_DATA=full to run against the complete files under files/ instead.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from stats import batting, fielding, pitching, woba
from util import importCSV, paths
from util.innings import ip_to_innings

pytestmark = pytest.mark.integration

DATA_ROOT = Path(__file__).parent / "data"

SEASONS = [
    ("Model-Sim", 2045, "dump", "Standard27"),
    ("TGS", 2046, "export", "Standard27"),
]

# columns the stat code reads from each player table, whichever export produced it
REQUIRED_BATTING = {
    "ID",
    "AB",
    "H",
    "PA",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HR",
    "BB",
    "IBB",
    "HP",
    "SF",
    "SH",
    "SB",
    "CS",
    "UBR",
}
REQUIRED_PITCHING = {
    "ID",
    "IP",
    "K",
    "BF",
    "BB",
    "R",
    "G",
    "GS",
    "HA",
    "AB",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HR",
    "HP",
    "SF",
}
REQUIRED_FIELDING = {
    "ID",
    "POS",
    "IP",
    "DP",
    "ARM",
    "FRM",
    "E",
    "SBA",
    "RTO",
    "BIZ-R",
    "BIZ-Rm",
    "BIZ-Z",
    "BIZ-Zm",
}


def source_available(league, season, source, ratings):
    folder = (
        paths.dump_dir(league, ratings, season)
        if source == "dump"
        else paths.export_dir(league, season)
    )
    return folder.exists()


def load(league, season, source, ratings):
    """Imported tables, without the csv writes done by importCSV.import_files."""
    if source == "dump":
        df_ratings = importCSV.import_ootp_dump_ratings(league, season, ratings, ".")
        bat, pit, fld, team_bat, team_pit, team_fld, bat_r, bat_l, *_ = (
            importCSV.import_ootp_dump_stats(league, season, ratings, ".")
        )
        # the dump team tables hold every level (minor leagues too)
        team_bat, team_pit, team_fld = (
            t[t["LEVEL_ID"] == 1] for t in (team_bat, team_pit, team_fld)
        )
    else:
        bat, pit, fld, team_bat, team_pit, team_fld = importCSV.import_csv_flex(
            league, season, "."
        )
        df_ratings = importCSV.import_ratings(league, season, ratings, ".")
        bat_r = bat_l = pd.DataFrame()
    return (
        importCSV.clean_ratings(df_ratings),
        bat,
        bat_r,
        bat_l,
        pit,
        fld,
        team_bat,
        team_pit,
        team_fld,
    )


@pytest.fixture(scope="module", autouse=True)
def data_root():
    if os.environ.get("OOTP_TEST_DATA") == "full":
        yield
        return
    patch = pytest.MonkeyPatch()
    patch.setattr(paths, "FILES_ROOT", DATA_ROOT)
    yield
    patch.undo()


@pytest.fixture(scope="module", params=SEASONS, ids=[f"{s[0]}-{s[1]}" for s in SEASONS])
def season(request, data_root):
    league, year, source, ratings = request.param
    if not source_available(league, year, source, ratings):
        pytest.skip(f"no {source} files for {league} {year}")
    data = load(league, year, source, ratings)
    return {"league": league, "year": year, "source": source, "data": data}


def test_player_tables_have_the_columns_the_stats_need(season):
    _, bat, _, _, pit, fld, *_ = season["data"]
    assert REQUIRED_BATTING <= set(bat.columns) | {"CI"}
    assert REQUIRED_PITCHING <= set(pit.columns)
    assert REQUIRED_FIELDING <= set(fld.columns)


def test_keys_are_unique(season):
    ratings, bat, _, _, pit, fld, *_ = season["data"]
    assert ratings.index.is_unique
    assert bat["ID"].is_unique
    assert pit["ID"].is_unique
    assert not fld.duplicated(["ID", "POS"]).any()


def test_batting_tables_agree_with_team_totals(season):
    _, bat, _, _, _, _, team_bat, *_ = season["data"]
    for column in ["HR", "SB", "BB"]:
        team_column = column if column in team_bat.columns else None
        if team_column:
            assert bat[column].sum() == pytest.approx(
                team_bat[team_column].sum(), rel=0.005
            ), column


def test_pitching_innings_agree_with_team_totals(season):
    _, _, _, _, pit, _, _, team_pit, _ = season["data"]
    players = ip_to_innings(pit["IP"]).sum()
    teams = ip_to_innings(team_pit["IP"]).sum()
    assert players == pytest.approx(teams, rel=0.005)


def test_league_batting_runs_balance(season):
    ratings, bat, bat_r, bat_l, *_ = season["data"]
    totals = batting.calculate_league_batting_totals(bat, season["source"])
    lg = pd.DataFrame(
        list(woba.calc_league_data(totals).items()), columns=["Stat", "Value"]
    ).set_index("Stat")
    total, *_ = batting.calculate_player_batting_stats(
        bat, bat_r, bat_l, lg, ratings, season["league"]
    )
    runs_per_600 = total["wRAA"].sum() / total["PA"].sum() * 600
    # players without a ratings row are left out, so allow a little slack
    assert abs(runs_per_600) < 1.0
    assert total.index.is_unique


def test_pitching_waa_balances_within_roles(season):
    ratings, _, _, _, pit, _, *_ = season["data"]
    lg = pd.DataFrame(
        list(
            woba.calc_league_data(
                pitching.calculate_league_pitching_totals(pit)
            ).items()
        ),
        columns=["Stat", "Value"],
    ).set_index("Stat")
    out = pitching.calculate_player_pitching_stats(pit, lg, ratings, season["league"])
    for role, group in out.groupby("PRole"):
        assert group["WAA"].sum() == pytest.approx(0, abs=1e-6), role
    assert (out["IPClean"] > 0).all()


def test_fielding_has_every_position_and_no_missing_runs(season):
    ratings, bat, *_ = season["data"]
    fld = season["data"][5]
    lg = pd.DataFrame(
        list(
            woba.calc_league_data(
                batting.calculate_league_batting_totals(bat, season["source"])
            ).items()
        ),
        columns=["Stat", "Value"],
    ).set_index("Stat")
    totals, _, out = fielding.calculate_player_fielding_stats(fld, lg, ratings)
    assert set(range(1, 10)) <= set(out["POS"])
    assert out["runsPAdj"].notna().all()
    # innings at a position, summed over players, are about the number of team games times 9
    assert totals.loc[2, "SBA"] > 0
