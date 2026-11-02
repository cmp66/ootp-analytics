import numpy as np
import pandas as pd
import pytest

from model import outlook_data as outlook
from scripts import build_outlook_tables, train_models
from tests.test_model_registry import SEASONS, multi_season_root


def frame(**columns):
    return pd.DataFrame(columns)


def test_tier_cutoffs_scale_with_the_number_of_teams():
    assert outlook.tier_cutoffs(30) == {"top10": 10, "top50": 50}
    assert outlook.tier_cutoffs(32) == {"top10": 11, "top50": 53}
    assert outlook.tier_cutoffs(1) == {"top10": 1, "top50": 2}  # never below one rank


def test_entry_route_from_the_draft_fields_and_age():
    rows = frame(
        picked_in_draft=[1, 1, 0, 0, np.nan],
        college=[1, 0, 0, 0, 0],
        Age=[22, 18, 16, 21, 17],
    )
    assert outlook.entry_route(rows).tolist() == [
        "college_draft",
        "hs_draft",
        "international",
        "other",
        "international",
    ]


def test_floors_are_multiples_of_five_except_the_pitch_count():
    assert outlook.floor_to_five(47.9) == 45 and outlook.floor_to_five(50) == 50
    assert outlook._floor("IF RNG", 38.0) == 35 and outlook._floor("PIT", 3.4) == 3


def test_rank_slots_gives_ranks_pools_and_tiers_per_slot():
    values = pd.concat(
        [
            frame(ID=range(100), slot="SS", value=np.arange(100.0)),
            frame(ID=range(100, 130), slot="C", value=np.arange(30.0)),
        ],
        ignore_index=True,
    )
    ranked = outlook.rank_slots(values, 30)
    ss = ranked[ranked["slot"] == "SS"].set_index("ID")
    assert ss.loc[99, "rank"] == 1 and ss.loc[0, "rank"] == 100 and (ss["pool"] == 100).all()
    assert (ss["tier"] == "top10").sum() == 10 and (ss["tier"] == "top50").sum() == 40
    assert ss.loc[99, "pct"] == 1.0 and ss.loc[0, "pct"] == pytest.approx(0.01)
    c = ranked[ranked["slot"] == "C"]
    assert (c["tier"] == "top10").sum() == 10 and (c["tier"] == "top50").sum() == 20  # the pool is smaller than 50
    bigger = outlook.rank_slots(values, 60)  # a 60-team league: the cutoffs double
    assert (bigger[bigger["slot"] == "SS"]["tier"] == "top10").sum() == 20


def test_ties_still_get_distinct_ranks():
    ranked = outlook.rank_slots(frame(ID=[1, 2, 3], slot="DH", value=[5.0, 5.0, 5.0]), 30)
    assert sorted(ranked["rank"]) == [1, 2, 3]


GATES = {
    "SS": {"IF RNG": 50, "IF ARM": 40, "pot": 60},
    "SP": {"STM": 35, "PIT": 3},
}


def test_gate_needs_every_key_rating_or_the_potential_of_the_position():
    state = frame(
        **{
            "IF RNG": [55, 55, 30, 30, np.nan],
            "IF ARM": [45, 30, 80, 80, 80],
            "SS Pot": [10, 10, 70, 50, 70],
        },
        index=pd.Index([1, 2, 3, 4, 5], name="ID"),
    )
    assert outlook.qualifies(state, "SS", GATES).tolist() == [True, False, True, False, True]
    # row 5 has no range rating but the potential reaches the minimum; without it a missing rating does not qualify
    assert not outlook.qualifies(state.assign(**{"SS Pot": 10}), "SS", GATES).iloc[4]


def test_slots_without_a_gate_accept_everybody_and_pitchers_use_stamina_and_pitches():
    state = frame(STM=[40, 30, 50], PIT=[3, 5, 2], index=pd.Index([1, 2, 3], name="ID"))
    assert outlook.qualifies(state, "SP", GATES).tolist() == [True, False, False]
    assert outlook.qualifies(state, "RP", GATES).all() and outlook.qualifies(state, "DH", GATES).all()


def test_gates_are_the_first_percentile_of_the_top_starters_floored():
    ratings = np.arange(30, 81, 1.0)  # 30..80
    samples = [
        {
            slot: frame(
                **{key: ratings for key in outlook.GATE_KEYS[slot]},
                **({f"{slot} Pot": ratings} if slot in outlook.POSITIONS.values() else {}),
            )
            for slot in outlook.GATE_KEYS
        }
    ]
    gates = outlook.gates_from_samples(samples)
    assert gates["SS"]["IF RNG"] == 30 and gates["SS"]["pot"] == 30  # 1 % of 30..80 is 30.5 -> 30
    assert gates["SP"]["PIT"] == 30  # integer floor for the pitch count
    assert set(gates) == set(outlook.GATE_KEYS)


def test_gates_round_trip_through_the_toml_and_keep_other_ratings_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook.toml")
    assert outlook.load_gates("Standard27") is None
    outlook.write_gates("Standard27", GATES, "measured on a test")
    outlook.write_gates("Standard26", {"SS": {"IF RNG": 45, "IF ARM": 35, "pot": 40}}, "measured on another test")
    assert outlook.load_gates("Standard27") == GATES
    assert outlook.load_gates("Standard26") == {"SS": {"IF RNG": 45, "IF ARM": 35, "pot": 40}}
    text = (tmp_path / "outlook.toml").read_text()
    assert '"IF RNG" = 50' in text and 'measured = "measured on a test"' in text
    assert 'measured = "measured on another test"' in text


def states(rows):
    return pd.DataFrame(rows, columns=["ID", "season", "Age", "picked_in_draft", "college"]).set_index("ID")


def test_stay_flags_are_nan_not_false_when_the_future_is_outside_the_table():
    table = states([(1, s, 20 + s, 0, 0) for s in (1, 2, 3)] + [(2, s, 20 + s, 0, 0) for s in (1, 2, 3, 4, 5)])
    flags = outlook.stay_flags(table[["season"]], last_season=5, horizons=(1, 3)).sort_index()
    assert flags.loc[(1, 2), "stay_h1"] == 1.0 and flags.loc[(1, 3), "stay_h1"] == 0.0  # he left after season 3
    assert flags.loc[(2, 5), "observed_h1"] == False and np.isnan(flags.loc[(2, 5), "stay_h1"])  # noqa: E712
    assert flags.loc[(2, 2), "stay_h3"] == 1.0 and flags.loc[(1, 2), "stay_h3"] == 0.0  # he is gone by season 5
    assert np.isnan(flags.loc[(2, 3), "stay_h3"]) and flags.loc[(2, 3), "observed_h3"] == False  # noqa: E712


def test_player_history_marks_players_present_in_the_first_season_as_truncated():
    table = states([(1, 10, 25, 0, 0), (1, 11, 26, 0, 0), (2, 11, 16, 0, 0), (2, 12, 17, 0, 0), (3, 12, 22, 1, 1)])
    history = outlook.player_history(table, first_season=10)
    assert history["left_truncated"].to_dict() == {1: True, 2: False, 3: False}
    assert history["entry_season"].to_dict() == {1: 10, 2: 11, 3: 12}
    assert history["entry_route"].to_dict() == {1: "other", 2: "international", 3: "college_draft"}
    assert history["last_season"].to_dict() == {1: 11, 2: 12, 3: 12}


def test_future_values_come_from_the_season_h_later_and_never_from_an_earlier_one():
    values = frame(
        ID=[1, 1, 1, 2, 2],
        slot="SS",
        season=[1, 2, 3, 1, 2],
        value=[10.0, 20.0, 30.0, 5.0, 6.0],
        rank=[3, 2, 1, 4, 5],
        tier=["rest", "top50", "top10", "rest", "rest"],
    )
    future = outlook.future_slot_values(values, 1, last_season=3).set_index(["ID", "season"])
    assert future.loc[(1, 1), "value_h1"] == 20.0 and future.loc[(1, 2), "tier_h1"] == "top10"
    assert future.loc[(2, 1), "value_h1"] == 6.0
    assert not future.loc[(1, 3), "observed_h1"] and np.isnan(future.loc[(1, 3), "value_h1"])  # beyond the table
    assert future.loc[(2, 2), "observed_h1"] and not future.loc[(2, 2), "in_pool_h1"]  # left or no longer qualifies
    assert len(future) == len(values)


def test_time_share_compares_with_the_90th_percentile_of_the_same_level_and_role():
    volumes = [100.0, 600.0, 650.0, 700.0, 20.0, 20.0]
    table = pd.DataFrame(
        {
            "volume": volumes,
            "batter": True,
            "primary_level": [1, 1, 1, 1, 1, 3],
            "starter": False,
        },
        index=pd.Index(range(6), name="ID"),
    )
    shares = outlook._with_time_share(table)["time_share"]
    reference = np.quantile([100.0, 600.0, 650.0, 700.0], 0.9)  # level 1, players with at least 50 PA
    assert shares.iloc[0] == pytest.approx(100.0 / reference) and shares.iloc[4] == pytest.approx(20.0 / reference)
    assert shares.iloc[3] == 1.0
    assert np.isnan(shares.iloc[5])  # nobody at level 3 reaches the minimum volume: no reference


def test_playing_time_of_the_fixture_dump_matches_its_career_stats():
    from tests.test_model_registry import Path  # the tests/data folder holds the fixture dump
    from util import paths

    monkey = pytest.MonkeyPatch()
    monkey.setattr(paths, "FILES_ROOT", Path(__file__).parent / "data")
    try:
        times = outlook.playing_time("Model-Sim", 2045)
        stats = pd.read_csv(paths.dump_dir("Model-Sim", "Standard27", 2045) / "players_career_batting_stats.csv")
        mlb = stats[(stats["year"] == 2045) & (stats["split_id"] == 1) & (stats["level_id"] == 1)]
    finally:
        monkey.undo()
    assert times.index.name == "ID" and times.index.is_unique
    assert times["time_share"].dropna().between(0, 1).all()
    assert times["mlb_pa"].sum() == mlb["pa"].sum()


@pytest.fixture
def built(tmp_path, monkeypatch):
    multi_season_root(tmp_path, monkeypatch)
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook.toml")
    shares = {t: {"L": 0.1, "N": 0.7, "H": 0.2} for t in outlook.formats.TRAITS}
    shares["Prone"] = {"Durable": 0.1, "Normal": 0.7, "Fragile": 0.2}
    monkeypatch.setattr(outlook, "export_trait_shares", lambda ratings: shares)
    assert train_models.main(["Model-Sim", "--start", "2046", "--stop", "2050", "--no-cv"]) == 0
    return outlook.build_tables("Model-Sim", SEASONS, log=lambda *_: None)


@pytest.mark.integration
def test_build_tables_on_the_fixture_seasons(built, tmp_path):
    players, values, gates = built["players"], built["values"], built["gates"]
    assert (tmp_path / "outlook.toml").exists() and set(gates) == set(outlook.GATE_KEYS)
    assert not values.duplicated(["season", "ID", "slot"]).any() and not players.duplicated(["season", "ID"]).any()
    every = built["slot_values"]
    assert len(every[every["qualified"]]) == len(values) and len(every) > len(values)
    assert not every.duplicated(["season", "ID", "slot"]).any()
    # ranks are 1..pool for each season and slot, and the tiers hold the cutoffs of the league's team count
    for (season, slot), group in values.groupby(["season", "slot"]):
        assert sorted(group["rank"]) == list(range(1, len(group) + 1))
        cutoffs = outlook.tier_cutoffs(int(players.loc[players["season"] == season, "teams"].iloc[0]))
        assert (group["tier"] == "top10").sum() == min(len(group), cutoffs["top10"])
    pitchers = set(players.loc[players["LPOS"].isin(outlook.PITCHER_LPOS), "ID"])
    assert set(values.loc[values["slot"].isin(outlook.PITCHER_SLOTS), "ID"]) <= pitchers
    assert not set(values.loc[values["slot"].isin(outlook.HITTER_SLOTS), "ID"]) & pitchers
    # DH has no gate: nearly every hitter of the season is ranked there (a few lack a feature)
    first_hitters = ~players.loc[players["season"] == SEASONS[0], "LPOS"].isin(outlook.PITCHER_LPOS)
    dh = values[(values["season"] == SEASONS[0]) & (values["slot"] == "DH")]
    assert 0.9 * first_hitters.sum() <= len(dh) <= first_hitters.sum()
    # the last season has no observed future; the first season has only truncated players
    last = players[players["season"] == SEASONS[-1]]
    assert last["stay_h1"].isna().all() and (~last["observed_h1"].astype(bool)).all()
    assert players[players["season"] == SEASONS[0]]["left_truncated"].all()
    assert set(players["WE_cat"].dropna()) <= {"L", "N", "H"} and set(players["Prone3_cat"].dropna()) <= {
        "Durable",
        "Normal",
        "Fragile",
    }
    assert outlook.load_trait_bounds("Standard27")
    assert players.loc[players["season"] == SEASONS[0], "stay_h1"].eq(1.0).mean() > 0.9


@pytest.mark.integration
def test_script_writes_the_tables_and_reports_bad_requests(built, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(outlook, "OUTLOOK_CONFIG", tmp_path / "outlook2.toml")
    assert build_outlook_tables.main(["Model-Sim", "--start", "2046", "--stop", "2048"]) == 0
    assert "player-seasons" in capsys.readouterr().out
    tables = outlook.read_tables("Model-Sim")
    assert set(tables["players"]["season"]) == {2046, 2047, 2048}
    assert build_outlook_tables.main(["Nope"]) == 2
    assert "error:" in capsys.readouterr().err
