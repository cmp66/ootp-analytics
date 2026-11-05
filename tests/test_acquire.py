import numpy as np
import pandas as pd
import pytest

from stats import finances
from viewer import acquire

SETTINGS = acquire.Settings(bullpen=2, hard_to_sign=("Impossible",))


def player(pid, org, slot, value, level="MLB", **extra):
    row = {
        "ID": pid, "slot": slot, "value_now": float(value), "org": org, "level_name": level, "team_level": level, "Age": 28,
        "First Name": "F", "Last Name": f"P{pid}", "is_free_agent": org == "FA", "WAIV": "-", "DEM": "0", "Sign": "Normal",
        "SLR": "$1 000 000", "YL": "1",
    }  # fmt: skip
    return {**row, **extra}


def league_frame():
    """Four organizations with a starter at every place (value 4 at each position, 0 for pitchers) except MINE, which has no
    catcher; a catcher on the free agent list (value 10, asks $2m), an expensive one (12, $30m), one nobody can sign (20,
    Impossible), one on waivers (8, $5m) and one of MINE's own in AAA (6)."""
    rows, pid = [], 0
    for org in ("MINE", "A", "B", "C"):
        for slot in acquire.LINEUP:
            if org == "MINE" and slot == "C":
                continue
            pid += 1
            rows.append(player(pid, org, slot, 4 + pid % 3))
        for n in range(5):
            pid += 1
            rows.append(player(pid, org, "SP", 0 + n))
        for n in range(2):
            pid += 1
            rows.append(player(pid, org, "RP", 0 + n))
    rows += [
        player(900, "FA", "C", 10, level="FA", DEM="$2.0m"),
        player(901, "FA", "C", 12, level="FA", DEM="$30.0m"),
        player(902, "FA", "C", 20, level="FA", DEM="$1.0m", Sign="Impossible"),
        player(903, "A", "C", 8, WAIV="Yes", SLR="$5 000 000"),
        player(904, "MINE", "C", 6, level="AAA"),
    ]
    return pd.DataFrame(rows)


def test_money_text_becomes_dollars():
    assert finances.parse_money("$201.9m") == 201_900_000 and finances.parse_money("$19 000 038") == 19_000_038
    assert finances.parse_money("$638k") == 638_000 and finances.parse_money("$0") == 0
    assert (
        np.isnan(finances.parse_money("-"))
        and np.isnan(finances.parse_money(None))
        and np.isnan(finances.parse_money("Free agent"))
    )


def test_finances_are_read_per_team_with_the_code_of_the_player_files(tmp_path, monkeypatch):
    from util import paths
    from util.config import SaveConfig

    folder = tmp_path / "finance"
    folder.mkdir()
    pd.DataFrame(
        {
            "ID": [1, 2], "Team Name": ["Cleveland Guardians", "Nowhere"], "Pay": ["$100.0m", "$1.0m"],
            "BgtSpc": ["$19 000 038", "$0"], "Mode": ["Win Now!", "Neutral"],
        }
    ).to_csv(folder / "fin.csv", index=False)  # fmt: skip
    monkeypatch.setattr(paths, "finance_dir", lambda save, season: folder)
    monkeypatch.setattr(
        finances, "get_save", lambda save: SaveConfig("S", "export", "R", team_names={"CLE": "Cleveland Guardians"})
    )
    table = finances.read_finances("S", 1)
    assert table["Team"].iloc[0] == "CLE" and pd.isna(
        table["Team"].iloc[1]
    )  # a team that config/saves.toml does not name
    assert (
        table.loc[0, "Payroll"] == 100_000_000
        and table.loc[0, "BudgetSpace"] == 19_000_038
        and table.loc[0, "Mode"] == "Win Now!"
    )
    monkeypatch.setattr(paths, "finance_dir", lambda save, season: tmp_path / "none")
    assert finances.read_finances("S", 1) is None


def test_a_place_nobody_can_play_is_a_need_with_nobody_in_it_and_the_places_are_ranked_by_the_gap():
    needs = acquire.needs(league_frame(), "MINE", SETTINGS)
    first = needs.iloc[0]
    assert first["place"] == "C" and first["Player"] == "(nobody)" and first["need"] and first["pct"] == 0
    assert len(needs) == len(acquire.places(SETTINGS)) and needs["place"].is_unique
    assert needs["gap"].iloc[0] >= needs["gap"].iloc[-1]


def test_candidates_come_from_my_own_non_starters_waivers_and_free_agents_with_what_each_costs():
    frame = league_frame()
    needs = acquire.needs(frame, "MINE", SETTINGS)
    pool = acquire.candidate_pool(frame, "MINE", needs, SETTINGS)
    catchers = pool[pool["slot"] == "C"].set_index("ID")
    assert set(catchers.index) == {900, 901, 903, 904}  # 902 asks the impossible and is left out
    assert catchers.loc[900, "Source"] == acquire.FREE_AGENT and catchers.loc[900, "Cost"] == 2_000_000
    assert (
        catchers.loc[903, "Source"] == acquire.WAIVERS
        and catchers.loc[903, "Cost"] == 5_000_000
        and catchers.loc[903, "From"] == "A"
    )
    assert (
        catchers.loc[904, "Source"] == acquire.INTERNAL
        and catchers.loc[904, "Cost"] == 0
        and catchers.loc[904, "From"] == "AAA"
    )
    assert not set(needs["ID"].dropna()) & set(pool["ID"])  # a starter is not a candidate for another place
    loose = acquire.candidate_pool(frame, "MINE", needs, acquire.Settings(hard_to_sign=()))
    assert 902 in set(loose["ID"])


def test_the_gain_is_the_value_over_the_starter_and_a_reliever_counts_for_less():
    frame = league_frame()
    needs = acquire.needs(frame, "MINE", SETTINGS)
    pool = acquire.candidate_pool(frame, "MINE", needs, SETTINGS)
    c = needs[needs["place"] == "C"].iloc[0]
    found = acquire.options(pool, c, SETTINGS).set_index("ID")
    assert found.loc[900, "Gain"] == pytest.approx(10 - c["value"]) and list(found.index)[0] == 901  # best gain first
    assert SETTINGS.weight("RP1") == 0.35 and SETTINGS.weight("SP3") == 1.0 and SETTINGS.weight("C") == 1.0


def test_the_best_package_stays_within_the_budget_and_uses_a_candidate_once():
    frame = league_frame()
    needs = acquire.needs(frame, "MINE", SETTINGS)
    pool = acquire.candidate_pool(frame, "MINE", needs, SETTINGS)
    rich = acquire.best_package(needs, pool, 100e6, SETTINGS)
    assert 901 in set(rich["ID"]) and 900 not in set(rich["ID"])  # one catcher, the best he can be
    assert rich["ID"].is_unique and rich["Place"].is_unique
    modest = acquire.best_package(needs, pool, 6e6, SETTINGS)
    assert modest["Cost"].sum() <= 6e6 and 901 not in set(modest["ID"]) and 900 in set(modest["ID"])  # the $2m catcher
    no_fa = pool[pool["Source"] != acquire.FREE_AGENT]
    claim = acquire.best_package(needs, no_fa, 6e6, SETTINGS)
    assert 903 in set(claim["ID"]) and claim["Cost"].sum() <= 6e6  # without free agents, the $5m waiver claim
    broke = acquire.best_package(needs, pool, 0, SETTINGS)
    assert broke["Cost"].sum() == 0 and set(broke["ID"]) == {904}  # only the player he already has


def test_budget_space_is_read_from_the_finances():
    table = pd.DataFrame({"Team": ["CLE", "NYY"], "BudgetSpace": [1.5e6, 9e6]})
    assert acquire.budget_space(table, "CLE") == 1.5e6 and np.isnan(acquire.budget_space(table, "XXX"))
    assert np.isnan(acquire.budget_space(None, "CLE"))


def test_settings_come_from_the_config_file(tmp_path):
    config = tmp_path / "a.toml"
    config.write_text('[rules]\nneed_percentile = 0.2\nhard_to_sign = ["Hard"]\n[weights]\nRP = 0.5\n')
    settings = acquire.load_settings(config)
    assert settings.need_percentile == 0.2 and settings.hard_to_sign == ("Hard",) and settings.weight("RP2") == 0.5
    assert acquire.load_settings(tmp_path / "missing.toml") == acquire.Settings()
    config.write_text("[rules]\nbudget = 1\n")
    with pytest.raises(ValueError):
        acquire.load_settings(config)


def priorities(prospect_later: float, budget: float = 1e9, settings: acquire.Settings = SETTINGS) -> pd.DataFrame:
    """The priority list of MINE for the league frame; its AAA catcher (6 now) is worth `prospect_later` later."""
    frame = league_frame()
    frame["value_h1"] = frame["value_h2"] = frame["value_now"]
    frame.loc[frame["ID"] == 904, ["value_h1", "value_h2"]] = prospect_later
    needs = acquire.needs(frame, "MINE", settings)
    pool = acquire.candidate_pool(frame, "MINE", needs, settings)
    return acquire.priority_list(frame, needs, pool, budget, settings)


def test_a_hole_at_the_place_makes_the_long_contract_the_priority():
    found = priorities(prospect_later=0.0).set_index("Place")
    assert (
        found.loc["C", "Verdict"] == "Hole" and found.loc["C", "Player"] == "F P901"
    )  # 12 for three years beats 10 for one
    assert found.loc["C", "PerYear"] == pytest.approx(found.loc["C", "Impact"] / 3)
    assert found.loc["C", "Years"] == 3 and found.loc["C", "Commit"] == 3 * found.loc["C", "Cost"]


def test_a_prospect_who_will_be_ready_makes_the_long_contract_a_bridge_for_a_short_one():
    found = priorities(prospect_later=20.0).set_index("Place")
    assert found.loc["C", "Verdict"] == "Bridge" and "P904" in found.loc["C", "Context"]
    assert found.loc["C", "Player"] == "F P900" and found.loc["C", "Years"] == 1  # the one year deal does not block him


def test_the_list_keeps_the_minimum_budget_skips_what_does_not_fit_and_discounts_the_free_agents_who_may_not_sign():
    found = priorities(0.0, budget=12.5e6, settings=acquire.Settings(bullpen=2, min_budget=10e6)).set_index("Place")
    assert found.loc["C", "Spend"] == "Within"  # $2m of $12.5m leaves the $10m
    assert found.loc["C", "Player"] == "F P900"  # the $30m deal does not fit the space and the next best is listed
    row = found.loc["C"]
    assert row["Chance"] == acquire.Settings().chance("Normal") and row["Expected"] == pytest.approx(
        row["Impact"] * 0.7
    )
    assert acquire.Settings().chance("unknown level") == 1.0


def test_local_popularity_boosts_a_player_much_more_than_national_popularity():
    rows = pd.DataFrame(
        {
            "Nat. Pop.": ["Unkn.", "Pop.", "V.Pop", "Ex.Pop.", "bogus"],
            "Loc. Pop.": ["V.Pop", "Pop.", "Unkn.", "Fair", None],
        }
    )
    found = acquire.popularity(rows)
    assert list(found["local"]) == ["vpop", "pop", "unkn", "fair", "unkn"]
    assert list(found["national"]) == ["unkn", "pop", "vpop", "expop", "unkn"]
    assert list(acquire.popularity(pd.DataFrame({"x": [1]})).iloc[0]) == [
        "unkn",
        "unkn",
    ]  # an export without the columns
    settings = acquire.load_settings()
    assert dict(settings.popularity_local)["expop"] > 4 * dict(settings.popularity_national)["expop"]

    def with_pop(local, national, **changes):
        frame = league_frame()
        frame["Nat. Pop."] = "Unkn."
        frame["Loc. Pop."] = "Unkn."
        frame.loc[frame["ID"] == 900, ["Loc. Pop.", "Nat. Pop."]] = [local, national]
        needs = acquire.needs(frame, "MINE", SETTINGS)
        pool = acquire.candidate_pool(frame, "MINE", needs, SETTINGS)
        found = acquire.priority_list(
            frame, needs, pool, 5e6, acquire.Settings(bullpen=2, min_budget=0, **changes)
        )  # the $30m one is out
        return found[found["Place"] == "C"].iloc[0]

    plain, local, national = with_pop("Unkn.", "Unkn."), with_pop("Ex.Pop.", "Unkn."), with_pop("Unkn.", "Ex.Pop.")
    assert plain["Player"] == "F P900" and plain["Boost"] == 1.0
    assert local["Boost"] == pytest.approx(1.20) and national["Boost"] == pytest.approx(1.04)
    assert with_pop("Ex.Pop.", "V.Pop")["Boost"] == pytest.approx(1.22)  # the two add up
    assert local["Expected"] == pytest.approx(plain["Expected"] * 1.20)
