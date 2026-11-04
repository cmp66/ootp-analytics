import numpy as np
import pandas as pd
import pytest

from viewer import minors, rule5

SETTINGS = rule5.Settings()


def players(**columns):
    """A table like rule5.player_table gives, one row per player, from lists in `columns` (the rest has defaults)."""
    n = len(next(iter(columns.values())))
    base = {
        "ID": list(range(1, n + 1)), "Player": [f"P{i}" for i in range(1, n + 1)], "Pos": ["SS"] * n, "Age": [23] * n,
        "Level": ["AAA"] * n, "pitcher": [False] * n, "Value": [-10.0] * n, "MLB": [0.5] * n, "Gain": [0.0] * n,
        "Upside": [-5.0] * n,
        "Ceil": [0.0] * n, "Floor": [-20.0] * n, "Top50": [1.0] * n, "ceiling_pct": [0.5] * n, "Tier": [minors.DEPTH] * n,
        "Window": [3] * n, "focus": [False] * n, "R5": [True] * n, "On40": [False] * n, "Owed": [750000.0] * n,
        "Years": ["1"] * n,
    }  # fmt: skip
    return pd.DataFrame({**base, **columns})


def test_the_export_flag_says_yes_with_yes_y_1_or_true_and_nothing_else():
    flags = rule5.flagged(pd.Series(["Yes", "yes", "Y", "1", True, "-", "No", "0", np.nan]))
    assert flags.tolist() == [True, True, True, True, True, False, False, False, False]


def test_the_40_man_comes_from_the_export_flag_or_is_estimated_without_it():
    frame = pd.DataFrame(
        {"ON40": ["Yes", "-", "0"], "level_name": ["AAA", "MLB", "AA"], "R5": ["No", "No", "Yes"], "SECY": [1, 0, 2]}
    )
    flags, exact = rule5.on_forty(frame, SETTINGS)
    assert exact and flags.tolist() == [True, False, False]
    flags, exact = rule5.on_forty(frame.drop(columns="ON40"), SETTINGS)
    assert not exact and flags.tolist() == [
        True,
        True,
        False,
    ]  # AAA with secondary time, the MLB player; not the draft eligible one
    assert rule5.on_forty(frame.drop(columns=["ON40", "SECY"]), SETTINGS)[0].tolist() == [False, True, False]


def test_protect_prospects_who_are_ready_and_players_who_would_start_and_expose_the_rest():
    table = players(
        Player=[
            "ready prospect",
            "top prospect",
            "unready prospect",
            "would start",
            "filler",
            "prospect with no upside",
        ],
        MLB=[0.4, 0.05, 0.05, 0.6, 0.5, 0.5],
        Tier=[minors.PROSPECT, minors.TOP_PROSPECT, minors.PROSPECT, minors.PAST, minors.DEPTH, minors.PROSPECT],
        focus=[True, True, True, False, False, True],
        Upside=[4.0, -3.0, 4.0, -1.0, -5.0, -2.0],
        Gain=[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    )
    rec = rule5.recommend(table, SETTINGS).set_index("Player")
    assert rec.loc["ready prospect", "Action"] == rule5.PROTECT and rec.loc["top prospect", "Action"] == rule5.PROTECT
    assert rec.loc["would start", "Action"] == rule5.PROTECT and "start" in rec.loc["would start", "Why"]
    assert rec.loc["unready prospect", "Action"] == rule5.LOW_RISK
    assert rec.loc["filler", "Action"] == rule5.EXPOSE and rec.loc["prospect with no upside", "Action"] == rule5.EXPOSE
    assert list(rec["Action"].map(rule5.ORDER)) == sorted(rec["Action"].map(rule5.ORDER))  # protect first
    assert rule5.recommend(table.assign(R5=False), SETTINGS).empty  # only the draft eligible are listed


def test_the_readiness_threshold_is_a_setting():
    table = players(Player=["a"], MLB=[0.3], Tier=[minors.PROSPECT], focus=[True], Upside=[2.0])
    assert rule5.recommend(table, SETTINGS)["Action"].iloc[0] == rule5.PROTECT
    assert rule5.recommend(table, rule5.Settings(ready_percentile=0.5))["Action"].iloc[0] == rule5.LOW_RISK


def test_the_40_man_arithmetic_counts_the_open_slots_the_protected_and_the_signings():
    table = players(
        On40=[True] * 36 + [False] * 2, Player=[f"x{i}" for i in range(38)], ID=list(range(38)), R5=[False] * 38
    )
    room = rule5.roster_room(table, SETTINGS, protect=3, signings=2)
    assert room == {"on_forty": 36, "open": 4, "needed": 5, "to_clear": 1}
    assert rule5.roster_room(table, SETTINGS, protect=1, signings=0)["to_clear"] == 0
    assert (
        rule5.roster_room(table, rule5.Settings(size=30), 0, 0)["open"] == 0
    )  # more on it than it holds: nothing open


def test_the_players_who_could_come_off_are_not_starters_and_the_least_worth_keeping_go_first():
    table = players(
        ID=[1, 2, 3, 4], Player=["starter", "prospect", "filler", "bench"], On40=[True, True, True, True], R5=[False] * 4,
        MLB=[0.9, 0.05, 0.02, 0.4], ceiling_pct=[0.1, 0.9, 0.1, 0.1], focus=[False, True, False, False],
        Value=[5.0, -20.0, -30.0, -3.0],
    )  # fmt: skip
    starters = pd.DataFrame({"org": ["X", "X"], "ID": [1, 99]})
    out = rule5.removal_candidates(table, starters, "X")
    assert list(out["Player"]) == ["filler", "bench", "prospect"] and 1 not in set(
        out["ID"]
    )  # the prospect is kept by his ceiling
    assert out["Keep"].is_monotonic_increasing


def test_the_settings_come_from_the_config_file(tmp_path):
    config = tmp_path / "r.toml"
    config.write_text('[roster]\nsize = 38\ncolumn = "X40"\n[rules]\nready_percentile = 0.3\n')
    assert rule5.load_settings(config) == rule5.Settings(size=38, column="X40", ready_percentile=0.3)
    assert rule5.load_settings(tmp_path / "missing.toml") == rule5.Settings()
    config.write_text("[rules]\nbudget = 1\n")
    with pytest.raises(ValueError):
        rule5.load_settings(config)
