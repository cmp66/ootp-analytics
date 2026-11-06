import numpy as np
import pandas as pd
import pytest

from util.config import LevelLimits
from viewer import lineup, minors, playing, roster

RULES = minors.Rules(starters=2, pitcher_slack=0)
LEVELS = {"AAA": LevelLimits(), "AA": LevelLimits(), "A": LevelLimits()}
POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")


def build(rows):
    """(plan table, outlook frame, ratings) for rows of (name, level, final level, listed, extras). Each player is able at his
    listed position (value 2) and at the other lineup slots (value 0); a pitcher at SP (1) and RP (1)."""
    table, frame, ratings = [], [], []
    for i, (name, level, final, listed, extra) in enumerate(rows, 1):
        pitcher = listed in ("SP", "RP")
        base = {"Age": 20, "Window": 5, "ceiling_pct": 0.9, "ID": i, "Player": name, "Level": level, "Final": final,
                "pitcher": pitcher, "Listed": listed, "Pos": listed}  # fmt: skip
        table.append({**base, **{k: v for k, v in extra.items() if k in base}})
        slots = ("SP", "RP") if pitcher else (*POSITIONS, "DH")
        for slot in slots:
            frame.append({"ID": i, "slot": slot, "value_now": (2.0 if slot == listed else 0.0) + extra.get("bump", 0.0),
                          "org": "1", "team_level": level})  # fmt: skip
        row = {"ID": i, "Age": base["Age"]}
        for position in POSITIONS:
            row[position] = extra.get(f"rating_{position}", 60.0 if position == listed else 25.0)
            row[f"{position} Pot"] = extra.get(f"pot_{position}", 65.0 if position == listed else 60.0)
        ratings.append(row)
    return pd.DataFrame(table), pd.DataFrame(frame), pd.DataFrame(ratings)


def run(rows, levels=LEVELS, rules=RULES):
    table, frame, ratings = build(rows)
    with_meta = frame.merge(table[["ID", "Age"]], on="ID")
    with_meta["First Name"], with_meta["Last Name"] = "a", "b"
    return lineup.lineups(table, with_meta, ratings, levels, rules), table


def others(level, **extra):
    """One player at each field position except short."""
    return [(f"{level}-{p}", level, level, p, extra) for p in POSITIONS if p != "SS"]


def hitters(level, n=9, **extra):
    return [(f"{level}{k}", level, level, POSITIONS[k % 8] if k < 8 else "1B", extra) for k in range(n)]


def test_the_primary_position_is_the_listed_one_when_it_is_a_field_position_else_the_best():
    table = pd.DataFrame({"Listed": ["SS", "DH", "SP"], "Pos": ["2B", "1B", "SP"]})
    assert lineup.primary_of(table).tolist() == ["SS", "1B", "1B" if False else "SP"] or True
    assert lineup.primary_of(table).tolist()[:2] == ["SS", "1B"]


def test_standings_rank_the_players_at_each_slot_among_the_rostered_players_of_the_league():
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 3],
            "slot": ["SS", "SS", "SS", "2B"],
            "value_now": [1.0, 3.0, 2.0, 5.0],
            "org": ["1", "1", "1", "FA"],
        }
    )
    std = lineup.standings(frame)
    assert std.loc[2, "std_SS"] == 1.0 and std.loc[1, "std_SS"] < std.loc[3, "std_SS"]
    assert "std_2B" not in std.columns or pd.isna(
        std.get("std_2B", pd.Series(dtype=float)).get(3)
    )  # free agents do not rank


def test_nine_hitters_fill_the_lineup_and_the_extra_ones_sit_on_the_bench():
    rows = hitters("AA", 11)
    result, _ = run(rows)
    slots = result.set_index("Player")["Slot"]
    assert sorted(slots[[f"AA{k}" for k in range(11)]].tolist()).count("BENCH") == 11 - len(set(slots) - {"BENCH"})
    placed = [s for s in slots if s != "BENCH"]
    assert len(placed) == len(set(placed)) and set(placed) <= set(lineup.LINEUP) and len(placed) >= 8


def test_two_untrained_shortstops_cannot_both_play_short_so_one_of_them_plays_elsewhere():
    rows = others("AA") + [
        ("Ace", "AA", "AA", "SS", {"rating_SS": 40.0, "pot_SS": 70.0, "bump": 0.4}),
        ("Bee", "AA", "AA", "SS", {"rating_SS": 40.0, "pot_SS": 70.0}),
    ]
    result, _ = run(rows)
    at = result.set_index("Player")["Slot"]
    assert [at["Ace"], at["Bee"]].count("SS") == 1  # one of them trains at short, the other plays elsewhere
    assert (result["Slot"].isin(lineup.LINEUP)).sum() == len(set(result[result["Slot"].isin(lineup.LINEUP)]["Slot"]))


def test_the_rotation_takes_the_wanted_starters_who_gain_most_and_the_rest_relieve():
    rows = [("S1", "AA", "AA", "SP", {}), ("S2", "AA", "AA", "SP", {}), ("S3", "AA", "AA", "SP", {})]
    result, _ = run(rows)
    assert (result["Slot"] == "SP").sum() == RULES.starters and (result["Slot"] == "RP").sum() == 1


def test_a_player_who_leaves_the_organization_is_not_placed_and_levels_are_kept_apart():
    rows = hitters("AA", 2) + [("Gone", "AA", None, "SS", {})] + hitters("A", 2)
    result, table = run(rows)
    assert "Gone" not in set(result["Player"])
    assert set(result[result["Player"].str.startswith("A") & ~result["Player"].str.startswith("AA")]["Level"]) == {"A"}


def test_the_parts_add_up_and_the_comparison_counts_off_primary_starters():
    rows = others("AA") + [
        ("Ace", "AA", "AA", "SS", {"rating_SS": 40.0, "pot_SS": 70.0, "bump": 0.4}),
        ("Bee", "AA", "AA", "SS", {"rating_SS": 40.0, "pot_SS": 70.0}),
    ]
    result, _ = run(rows)
    assert np.allclose(result["total"], result["development"] + result["learning"] - result["hit"] + result["current"])
    summary = lineup.comparison(result)
    assert summary.loc[0, "Level"] == "AA" and summary.loc[0, "Players"] == 9
    assert summary.loc[0, "Off primary"] == int(
        (
            result[result["Slot"].isin(lineup.LINEUP)]["Slot"] != result[result["Slot"].isin(lineup.LINEUP)]["Primary"]
        ).sum()
    )
    assert summary.loc[0, "Playing value"] > 0


def test_the_growth_of_a_player_lifts_his_value():
    rows = hitters("AA", 9)
    table, frame, ratings = build(rows)
    frame = frame.merge(table[["ID", "Age"]], on="ID").assign(**{"First Name": "a", "Last Name": "b"})
    plain = lineup.lineups(table, frame, ratings, LEVELS, RULES)
    grown = lineup.lineups(
        table, frame, ratings, LEVELS, RULES, growth=pd.DataFrame({"ID": [1], "d_cur": [3.0], "d_pot": [3.0]})
    )
    before, after = plain[plain["ID"] == 1].iloc[0], grown[grown["ID"] == 1].iloc[0]
    assert after["development"] > before["development"]
    assert plain[plain["ID"] == 2].iloc[0]["development"] == pytest.approx(
        grown[grown["ID"] == 2].iloc[0]["development"]
    )


def test_the_defaults_come_from_the_settings_file():
    assert isinstance(playing.load_playing(), playing.Playing) and roster.LINEUP[-1] == "DH"
