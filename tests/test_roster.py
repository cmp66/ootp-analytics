from unittest import mock

import numpy as np
import pandas as pd
import pytest

from util.config import LevelLimits
from viewer import minors, roster

RULES = minors.Rules(starters=1, pitcher_slack=0, position_gap=2.0)


def table(rows):
    """A table like minors.manage gives, from (Level, pitcher, extras) rows (the rest has defaults: a young depth player who
    stays)."""
    base = {
        "Age": 22, "Pro": 2, "Move": "", "To": "", "Blocked": "", "Reason": "", "focus": False, "Tier": minors.DEPTH,
        "Window": 3, "ceiling_pct": 0.3, "mean": 0.5, "next_mean": 0.3,
        "green": 0, "Value": 0.0, "Ceil": 0.0, "Floor": -9.0, "Top50": 1.0,
        "ON40": "No", "Listed": "SS",
    }  # fmt: skip
    out = []
    for i, (level, pitcher, extra) in enumerate(rows, 1):
        out.append({**base, "ID": i, "Player": f"P{i}", "Level": level, "pitcher": pitcher, "Pos": "RP" if pitcher else "SS",
                    "Listed": "RP" if pitcher else "DH", **extra})  # fmt: skip
    return pd.DataFrame(out)


def frame_of(table_, values=None):
    """The outlook frame of a table: a player is able at his listed position (value 1) and, listed elsewhere, a shortstop of
    value 0."""
    listed = dict(zip(table_["ID"], table_["Listed"])) if "Listed" in table_ else {}
    rows = [
        {"ID": i, "slot": listed.get(i) if listed.get(i) in (*roster.HITTER_POSITIONS, "DH") else "SS",
         "value_now": 1.0 if listed.get(i) in (*roster.HITTER_POSITIONS, "DH") else 0.0, "LPOS": "SS"}
        for i in table_["ID"] if i not in (values or {})
    ]  # fmt: skip
    rows += [
        {"ID": i, "slot": s, "value_now": v, "LPOS": "SS"} for i, per in (values or {}).items() for s, v in per.items()
    ]
    return pd.DataFrame(rows)


def run(rows, levels, values=None, rules=RULES):
    t = table(rows)
    return roster.plan(t, frame_of(t, values), levels, rules)


LEVELS = {"AAA": LevelLimits(), "AA": LevelLimits(max_players=2), "A": LevelLimits(max_age=25, max_players=3)}


def test_a_level_over_its_maximum_sends_its_fillers_down_and_keeps_the_prospect():
    prospect = {"focus": True, "Tier": minors.PROSPECT, "ceiling_pct": 0.9}
    plan = run(
        [("AA", False, prospect), ("AA", False, {}), ("AA", False, {"ceiling_pct": 0.1})], LEVELS
    ).players.set_index("Player")
    assert plan.loc["P1", "Action"] == "" and plan.loc["P2", "Action"] != "" or plan.loc["P3", "Action"] != ""
    assert (plan["Action"] == roster.DEMOTE).sum() == 1 and (
        plan["Final"] == "AA"
    ).sum() == 2  # one down, the maximum kept
    assert (
        plan.loc["P3", "Action"] == roster.DEMOTE and "holds at most 2" in plan.loc["P3", "Why"]
    )  # the weaker depth player


def test_a_player_too_old_for_the_level_below_is_released_not_demoted_and_the_40_man_decides_dfa_or_release():
    levels = {
        "AAA": LevelLimits(max_players=1),
        "AA": LevelLimits(max_players=2),
        "A": LevelLimits(max_age=25, max_players=3),
    }
    prospect = {"ceiling_pct": 0.9, "focus": True, "Tier": minors.PROSPECT}
    old = {"Age": 26, "ceiling_pct": 0.05, "Window": 0}
    rows = [("AAA", False, prospect), ("AA", False, prospect), ("AA", False, old), ("AA", False, prospect)]
    plan = run(rows, levels).players.set_index("Player")
    assert plan.loc["P3", "Action"] == roster.RELEASE and pd.isna(plan.loc["P3", "Final"])  # A takes nobody over 25
    assert "no level below takes him" in plan.loc["P3", "Why"] and plan.loc[["P2", "P4"], "Action"].eq("").all()
    rows[2] = ("AA", False, {**old, "ON40": "Yes"})
    assert run(rows, levels).players.set_index("Player").loc["P3", "Action"] == roster.DFA
    t = table(rows).drop(columns="ON40")  # a league without the flag: the page cannot tell
    assert (
        roster.plan(t, frame_of(t), levels, RULES).players.set_index("Player").loc["P3", "Action"]
        == roster.DFA_OR_RELEASE
    )


def test_a_player_over_the_age_limit_of_his_level_goes_up_or_out_never_stays():
    levels = {"AAA": LevelLimits(max_players=1), "A": LevelLimits(max_age=25, max_players=3)}
    rows = [("A", False, {"Age": 26, "Move": minors.MUST, "To": "AAA", "Reason": "over the limit"}), ("AAA", False, {})]
    plan = run(rows, levels).players.set_index("Player")
    assert plan.loc["P1", "Final"] != "A" and plan.loc["P1", "Action"] in (roster.PROMOTE, roster.RELEASE, roster.DFA)
    assert (plan["Final"] == "AAA").sum() <= 1 or plan.loc[
        "P1", "Action"
    ] != roster.PROMOTE  # AAA is full: the other goes down


def test_the_pitchers_of_a_level_stay_within_the_slack_of_the_target_when_players_have_to_go():
    levels = {"AA": LevelLimits(max_players=3, pitchers=2)}
    rows = [("AA", True, {"ceiling_pct": 0.2}), ("AA", True, {"ceiling_pct": 0.25}), ("AA", False, {"ceiling_pct": 0.5}),
            ("AA", False, {"ceiling_pct": 0.6})]  # fmt: skip
    strict = run(rows, levels).players.set_index("Player")  # slack 0: two pitchers stay, so a hitter goes
    assert strict.loc["P3", "Action"] == roster.RELEASE and strict.loc[["P1", "P2"], "Action"].eq("").all()
    loose = run(rows, levels, rules=minors.Rules(starters=1, pitcher_slack=1)).players.set_index(
        "Player"
    )  # one fewer is allowed
    assert loose.loc["P1", "Action"] == roster.RELEASE  # the least worth keeping goes


def test_nothing_moves_when_the_levels_are_within_their_limits():
    plan = run([("AA", False, {}), ("A", False, {})], LEVELS)
    assert (plan.players["Action"] == "").all() and plan.positions.empty
    assert list(plan.summary["After"]) == [1, 1] and list(plan.summary["Max"]) == [2, 3]


def test_the_plan_never_breaks_a_limit_it_can_keep():
    rng = np.random.default_rng(1)
    levels = {
        "AAA": LevelLimits(max_players=6),
        "AA": LevelLimits(max_players=6),
        "A": LevelLimits(max_age=25, max_players=8),
    }
    rows = []
    for _ in range(26):
        level = rng.choice(["AAA", "AA", "A"])
        age = int(rng.integers(20, 28)) if level != "A" else int(rng.integers(20, 27))
        rows.append((level, bool(rng.integers(0, 2)), {"Age": age, "ceiling_pct": float(rng.random()),
                    "focus": bool(rng.integers(0, 2)), "Move": str(rng.choice(["", "", minors.UP, minors.DOWN]))}))  # fmt: skip
    t = table(rows)
    t["To"] = t["Move"].map({minors.UP: "up", minors.DOWN: "down"}).fillna("")
    plan = roster.plan(t, frame_of(t), levels, RULES).players
    kept = plan[plan["Final"].notna() & (plan["Final"] != "MLB")]
    for level, limits in levels.items():
        assert (kept["Final"] == level).sum() <= limits.max_players
    assert not ((kept["Final"] == "A") & (kept["Age"] > 25)).any()  # nobody over the limit of A is at A
    moved = plan[plan["Action"].isin([roster.PROMOTE, roster.DEMOTE])]
    order = ["MLB", "AAA", "AA", "A"]
    assert all(
        abs(order.index(a) - order.index(b)) == 1 for a, b in zip(moved["Level"], moved["Final"])
    )  # one level at a time


def test_a_hitter_the_best_lineup_plays_elsewhere_is_flagged_and_so_are_relievers_who_belong_in_the_rotation():
    levels = {"AA": LevelLimits()}
    rows = [
        ("AA", False, {"Listed": "SS"}),
        ("AA", False, {"Listed": "SS"}),
        ("AA", True, {"Listed": "RP"}),
        ("AA", True, {"Listed": "SP"}),
    ]
    t = table(rows)
    values = {
        1: {"SS": 10.0, "3B": 14.0},
        2: {"SS": 12.0, "3B": 1.0},  # the best lineup: 1 at 3B (+4 over SS), 2 at SS
        3: {"SP": 5.0},
        4: {"SP": 1.0},
    }
    plan = roster.plan(t, frame_of(t, values), levels, RULES)
    found = plan.positions.set_index("Player")
    assert (
        found.loc["P1", "Better"] == "3B" and found.loc["P1", "Gain"] == pytest.approx(4.0) and "P2" not in found.index
    )
    assert found.loc["P3", "Better"] == "SP" and found.loc["P4", "Better"] == "RP"  # one starter wanted: the better one
    assert plan.summary.loc["AA", "Listed SP"] == 1 and plan.summary.loc["AA", "SP after"] == 1
    smaller_gap = roster.plan(t, frame_of(t, values), levels, minors.Rules(starters=1, position_gap=5.0)).positions
    assert "P1" not in set(smaller_gap["Player"])  # a gap of 4 runs is below the threshold of 5


def test_a_level_short_of_pitchers_gets_one_from_the_level_next_to_it_but_not_a_prospect():
    levels = {"AAA": LevelLimits(pitchers=1), "AA": LevelLimits(pitchers=2)}
    rows = [("AAA", False, {}), ("AA", True, {}), ("AA", True, {}), ("AA", True, {}), ("AA", False, {})]
    plan = run(rows, levels).players
    moved = plan[plan["Action"] == roster.PROMOTE]
    assert (
        len(moved) == 1 and bool(moved["pitcher"].iloc[0]) and moved["Level"].iloc[0] == "AA"
    )  # three pitchers, target two
    assert "roster balance" in moved["Why"].iloc[0]
    prospects = [("AAA", False, {})] + [("AA", True, {"focus": True, "Tier": minors.PROSPECT, "ceiling_pct": 0.95})] * 3
    assert (run(prospects + [("AA", False, {})], levels).players["Action"] == "").all()  # not moved for the count
    inside = run(
        [("AAA", True, {}), ("AA", True, {}), ("AA", True, {})], levels
    )  # within one of each target: nothing to fix
    assert (inside.players["Action"] == "").all()


def test_at_the_top_level_a_player_close_to_the_majors_is_kept_and_further_down_the_prospects_count_more():
    levels = {"AAA": LevelLimits(max_players=2), "AA": LevelLimits()}
    near = {"next_mean": 0.8, "ceiling_pct": 0.1, "Window": 0}
    far = {"next_mean": 0.1, "ceiling_pct": 0.1, "Window": 0, "mean": 0.4}
    plan = run([("AAA", False, near), ("AAA", False, far), ("AAA", False, {**near, "next_mean": 0.7})], levels).players
    assert plan.set_index("Player").loc["P2", "Action"] == roster.DEMOTE and (plan["Final"] == "AAA").sum() == 2
    rules = minors.Rules()
    prospect = pd.Series({"Window": 3, "ceiling_pct": 0.9, "next_mean": 0.3, "mean": 0.5})
    assert roster._keep(prospect, 4, rules) > roster._keep(prospect, 0, rules)  # a prospect matters more further down
    filler = pd.Series({"Window": 0, "ceiling_pct": 0.1, "next_mean": 0.8, "mean": 0.5})
    assert (
        roster._keep(filler, 0, rules) > roster._keep(filler, 2, rules) > roster._keep(filler, 4, rules) == 1.0
    )  # filler is filler


def test_nobody_is_demoted_into_a_level_that_players_are_placed_in_automatically():
    levels = {"AA": LevelLimits(max_players=1), "INT": LevelLimits(max_players=5, no_demotion=True)}
    rows = [("AA", False, {}), ("AA", False, {"ceiling_pct": 0.1}), ("INT", False, {})]
    plan = run(rows, levels).players.set_index("Player")
    assert plan.loc["P2", "Action"] == roster.RELEASE  # AA is over its maximum and INT takes nobody from above
    assert (
        not (plan["Final"] == "INT").any() or plan.loc["P3", "Final"] == "INT"
    )  # only the player already there is in it
    assert plan.loc["P3", "Action"] in ("", roster.PROMOTE)  # and he can be promoted out of it


PEERS = pd.DataFrame(
    {"pitcher": False, "value_now": np.arange(0.0, 100.0)}
)  # the league's major leaguers: value 0 to 99


def with_majors(minor_rows, mlb_rows, limit, rules=RULES):
    """Plan for minor league rows plus the active roster of the majors (rows of Level MLB). A player's `Value` is where he
    stands among the league's major leaguers: 90 is the 90th percentile."""
    levels = {"AAA": LevelLimits(max_players=10), "AA": LevelLimits(max_players=10)}
    t = table(mlb_rows + minor_rows)
    mlb, minors_ = t[t["Level"] == "MLB"], t[t["Level"] != "MLB"]
    with mock.patch.object(
        roster, "BAT_ONLY_MAX", 99
    ):  # the fillers are all DH: the bat-only limit is tested on its own
        return roster.plan(minors_, frame_of(t), levels, rules, mlb, limit, PEERS).players.set_index("Player")


def test_the_roster_limit_of_the_phase_sends_the_weakest_major_leaguers_down_and_a_larger_limit_keeps_everyone():
    mlb = [("MLB", False, {"Value": 90.0, "Tier": "MLB"}), ("MLB", False, {"Value": 10.0, "Tier": "MLB"}),
           ("MLB", False, {"Value": 50.0, "Tier": "MLB"})]  # fmt: skip
    cut = with_majors([("AAA", False, {})], mlb, 2)
    assert cut.loc["P2", "Action"] == roster.DEMOTE and cut.loc["P2", "Final"] == "AAA"  # the weakest goes down
    assert "roster limit of 2" in cut.loc["P2", "Why"]
    assert (with_majors([("AAA", False, {})], mlb, 3)["Action"] == "").all()
    assert (with_majors([("AAA", False, {})], mlb, 40)["Action"] == "").all()


def test_the_best_players_play_in_the_majors_and_one_without_an_option_left_cannot_be_sent_down_to_make_room():
    better = {"Value": 90.0}
    mlb = [
        ("MLB", False, {"Value": 50.0, "Tier": "MLB", "OPT": 1, "OY": 0}),
        ("MLB", False, {"Value": 95.0, "Tier": "MLB"}),
    ]
    swap = with_majors([("AAA", False, better)], mlb, 2)  # the 90 takes the spot of the 50, who has an option year
    assert swap.loc["P1", "Action"] == roster.DEMOTE and swap.loc["P3", "Action"] == roster.PROMOTE
    assert "uses option year 2 of 3" in swap.loc["P1", "Why"] and swap.loc["P3", "Final"] == "MLB"
    stuck = [("MLB", False, {"Value": 50.0, "Tier": "MLB", "OPT": 3, "OY": 0}), mlb[1]]  # all three option years used
    held = with_majors([("AAA", False, better)], stuck, 2)
    assert (
        held.loc["P1", "Action"] == "" and held.loc["P3", "Action"] == ""
    )  # waivers for a 40 point gain are not worth it
    last_year = [
        ("MLB", False, {"Value": 50.0, "Tier": "MLB", "OPT": 3, "OY": 2}),
        mlb[1],
    ]  # optioned already this season
    assert with_majors([("AAA", False, better)], last_year, 2).loc["P1", "Action"] == roster.DEMOTE
    nothing = with_majors([("AAA", False, {"Value": 20.0})], mlb, 2)  # a player below the floor does not take a spot
    assert (nothing["Action"] == "").all()


def test_the_options_of_a_player_follow_the_option_years_and_the_demotions_of_the_season():
    rules = minors.Rules()

    def can(**values):
        return roster.can_option(pd.Series(values), rules)

    assert (
        can(OPT=0, OY=0) and can(OPT=2, OY=0) and not can(OPT=3, OY=0)
    )  # three option years; the last is used at OPT 3
    assert can(OPT=3, OY=1) and can(OPT=3, OY=2)  # optioned this season: down and up as often as the season lasts
    assert can(OPT=float("nan"), OY=0) and can(OY=0) and can()  # no information: no restriction
    assert roster.option_text(pd.Series({"OPT": 3, "OY": 0}), rules).endswith("waivers")
    assert roster.option_text(pd.Series({"OPT": 1, "OY": 0}), rules) == "uses option year 2 of 3"
    assert roster.option_text(pd.Series({"x": 1}), rules) == ""


def test_a_problem_at_aaa_is_solved_from_below_and_never_by_sending_a_major_leaguer_down():
    levels = {"AAA": LevelLimits(max_players=10, pitchers=2), "AA": LevelLimits(max_players=10)}
    mlb = [
        ("MLB", True, {"Value": 20.0, "Tier": "MLB"}),
        ("MLB", False, {"Value": 60.0, "Tier": "MLB"}),
    ]  # a weak spare pitcher
    farm = [
        ("AAA", False, {}),
        ("AAA", False, {}),
        ("AA", True, {"ceiling_pct": 0.2}),
        ("AA", True, {"ceiling_pct": 0.1}),
    ]
    t = table(mlb + farm)
    plan = roster.plan(t[t["Level"] != "MLB"], frame_of(t), levels, RULES, t[t["Level"] == "MLB"], 10, PEERS).players
    assert (plan.loc[plan["Level"] == "MLB", "Action"] == "").all()  # nobody leaves the majors for the sake of AAA
    assert (
        plan.loc[plan["Level"].eq("AA") & plan["pitcher"] & plan["Action"].eq(roster.PROMOTE), "Final"].eq("AAA").sum()
        == 2
    )


def test_the_season_wants_13_pitchers_on_the_active_roster_and_the_other_phases_have_no_target():
    rules = minors.Rules()
    assert roster.mlb_limits(rules, 26) == LevelLimits(max_players=26, pitchers=13)
    assert roster.mlb_limits(rules, 40) == LevelLimits(max_players=40, pitchers=0)
    assert rules.mlb_starters == 5 and rules.max_option_years == 3
    assert minors.load_rules().mlb_spring == 60


def test_the_phase_is_read_from_the_active_rosters_of_the_league():
    rows = [{"ID": i, "team_level": "MLB", "org": f"T{i % 4}", "ACT": "Yes"} for i in range(4 * 40)]
    rules = minors.Rules()
    assert roster.detect_limit(pd.DataFrame(rows), rules) == 40
    assert roster.detect_limit(pd.DataFrame(rows[: 4 * 27]), rules) == 26
    assert (
        roster.detect_limit(pd.DataFrame({"team_level": ["MLB"], "org": ["A"]}), rules) is None
    )  # no ACT flag in the export


def test_every_level_keeps_the_players_it_needs_to_field_a_team_and_the_majors_can_pull_one_up():
    def hitters(level, listed):
        return [(level, False, {"Listed": position}) for position in listed]

    levels = {"AAA": LevelLimits(max_players=30), "AA": LevelLimits(max_players=30)}
    # the majors: 2 C, 6 IF and 5 OF (one spare each of IF and OF) and 5 starters; AAA has what it needs and a spare catcher
    team = (hitters("MLB", ["C", "C", "1B", "2B", "3B", "SS", "SS", "SS", "LF", "CF", "RF", "LF", "RF"]) +
            [("MLB", True, {"Listed": "SP"})] * 5)  # fmt: skip
    farm = (
        hitters("AAA", ["C", "C", "C", "1B", "2B", "3B", "SS", "SS", "LF", "CF", "RF", "LF"])
        + [("AAA", True, {"Listed": "SP"})] * 6
    )
    t = table([(lv, p, {**x, "mean": 0.5, "Tier": "MLB"}) for lv, p, x in team] + farm)
    mlb, rest = t[t["Level"] == "MLB"], t[t["Level"] != "MLB"]
    starters = pd.DataFrame({"ID": t.loc[t["Listed"] == "SP", "ID"], "slot": "SP", "value_now": 1.0, "LPOS": "SP"})
    frame = pd.concat([frame_of(t), starters])
    assert (roster.plan(rest, frame, levels, RULES, mlb, 18).players["Action"] == "").all()  # all there: nothing to do
    cut = roster.plan(rest, frame, levels, RULES, mlb, 17).players  # one too many: the one who goes is a spare
    gone = cut[cut["Level"].eq("MLB") & (cut["Action"] != "")]
    assert len(gone) == 1 and gone["Listed"].iloc[0] in ("1B", "2B", "3B", "SS", "LF", "CF", "RF")
    kept = cut[cut["Final"].eq("MLB")]["Listed"].value_counts()
    assert kept["C"] >= 2 and kept["SP"] >= 5 and sum(kept.get(p, 0) for p in ("1B", "2B", "3B", "SS")) >= 5
    assert sum(kept.get(p, 0) for p in ("LF", "CF", "RF")) >= 4
    short = t[t["ID"] != 2]  # the majors lose a catcher: the AAA catcher is pulled up
    mlb, rest = short[short["Level"] == "MLB"], short[short["Level"] != "MLB"]
    pulled = roster.plan(rest, pd.concat([frame_of(short), starters]), levels, RULES, mlb, 18).players
    assert (pulled["Final"].eq("MLB") & pulled["Level"].eq("AAA") & pulled["Listed"].eq("C")).sum() == 1


def test_the_summary_counts_what_each_level_has_after_the_plan_of_each_group_it_needs():
    rows = [("AA", False, {"Listed": "C"}), ("AA", False, {"Listed": "SS"}), ("AA", True, {"Listed": "SP"})]
    t = table(rows)
    starters = pd.DataFrame({"ID": [3], "slot": "SP", "value_now": 1.0, "LPOS": "SP"})
    plan = roster.plan(t, pd.concat([frame_of(t), starters]), {"AA": LevelLimits()}, RULES)
    row = plan.summary.loc["AA"]
    assert (row["Catchers"], row["Infielders"], row["Outfielders"]) == (1, 1, 0) and row["Can start"] == 1


def test_a_first_baseman_is_not_one_of_the_infielders_a_team_needs():
    def player(listed, value):
        return ("MLB", False, {"Listed": listed, "Value": value, "Tier": "MLB"})

    infield = [player("2B", 70), player("3B", 70), player("SS", 70), player("2B", 70), player("SS", 10)]
    mlb = [player("C", 70), player("C", 70), player("1B", 50), *infield] + [
        player(p, 70) for p in ("LF", "CF", "RF", "LF")
    ]
    plan = with_majors([("AAA", False, {"Listed": "1B", "Value": 99.0})], mlb, 12)
    assert (
        plan.loc["P7", "Action"] == ""
    )  # the weak shortstop is the fifth infielder: a first baseman would not replace him
    assert plan.loc["P3", "Action"] == roster.DEMOTE and plan.loc["P13", "Action"] == roster.PROMOTE


def test_an_injured_player_is_not_moved_either_way():
    well = {"INJ": "-", "Left": "-"}
    hurt = {"INJ": "Yes", "Left": "12 months"}
    mlb = [
        ("MLB", False, {"Value": 90.0, "Tier": "MLB", **well}),
        ("MLB", False, {"Value": 10.0, "Tier": "MLB", **hurt}),
    ]
    farm = [("AAA", False, {"Value": 99.0, **hurt}), ("AAA", False, {"Value": 80.0, **well})]
    plan = with_majors(farm, mlb, 2)
    assert (plan["Action"] == "").all()  # the hurt 99 is not promoted and the hurt 10 is not sent down
    assert plan.loc["P3", "Why"] == "injured (12 months): not moved until he is well"
    healthy = with_majors(
        [("AAA", False, {"Value": 99.0, **well})], mlb[:1] + [("MLB", False, {"Value": 10.0, "Tier": "MLB", **well})], 2
    )
    assert healthy.loc["P2", "Action"] == roster.DEMOTE and healthy.loc["P3", "Action"] == roster.PROMOTE


def majors_plan(mlb_rows, farm_rows, limit, bats=None):
    """`bats`: ID -> 1 (right), 2 (left), 3 (switch) for the `B` column of the outlook frame."""
    levels = {"AAA": LevelLimits(max_players=10), "AA": LevelLimits(max_players=10)}
    t = table(mlb_rows + farm_rows)
    mlb, rest = t[t["Level"] == "MLB"], t[t["Level"] != "MLB"]
    frame = frame_of(t)
    if bats:
        frame["B"] = frame["ID"].map(bats).fillna(1).astype(int)
    return roster.plan(rest, frame, levels, RULES, mlb, limit, PEERS).players.set_index("Player")


def test_the_capability_is_what_a_player_can_defend_not_the_position_the_game_lists():
    t = table([("MLB", False, {"Listed": "LF"}), ("MLB", False, {"Listed": "1B"}), ("MLB", False, {"Listed": "2B"}),
               ("MLB", False, {"Listed": "C"}), ("MLB", True, {})])  # fmt: skip
    frame = pd.DataFrame(
        [
            {"ID": 1, "slot": "LF", "value_now": -3.0}, {"ID": 1, "slot": "1B", "value_now": 2.0},  # listed LF, cannot field
            {"ID": 2, "slot": "1B", "value_now": 30.0}, {"ID": 2, "slot": "RF", "value_now": 8.0},  # a bat who could play RF
            {"ID": 3, "slot": "2B", "value_now": 4.0}, {"ID": 3, "slot": "SS", "value_now": 1.0},
            {"ID": 3, "slot": "3B", "value_now": -1.0}, {"ID": 4, "slot": "C", "value_now": -12.0},
        ]
    )  # fmt: skip
    cover = roster.capability(t, frame)
    assert list(cover["OF"]) == [
        False,
        False,
        False,
        False,
        False,
    ]  # the bat first player does not count as an outfielder
    assert list(cover["IF"]) == [False, False, True, False, False] and list(cover["C"]) == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert list(cover["BAT"]) == [True, True, False, False, False]  # the pitcher is not a hitter


def test_the_majors_carry_no_more_bat_only_hitters_than_a_first_baseman_and_a_platoon_at_dh():
    def fielders(position, n, value=70.0):
        return [("MLB", False, {"Listed": position, "Value": value, "Tier": "MLB"}) for _ in range(n)]

    mlb = (fielders("C", 2) + fielders("SS", 5) + fielders("LF", 4) + fielders("RF", 1, 80.0)  # a spare outfielder
           + fielders("DH", 3, 85.0))  # fmt: skip
    farm = [("AAA", False, {"Listed": "DH", "Value": 87.0})]
    plan = majors_plan(mlb, farm, 15)
    assert (plan["Action"] == "").all()  # a fourth bat does not take the place of a spare fielder
    with mock.patch.object(roster, "BAT_ONLY_MAX", 99):
        free = majors_plan(mlb, farm, 15)
    assert (
        free.loc["P16", "Action"] == roster.PROMOTE and (free.loc["P8":"P12", "Action"] == roster.DEMOTE).sum() == 1
    )  # without the limit
    weak = [*mlb[:-1], ("MLB", False, {"Listed": "DH", "Value": 20.0, "Tier": "MLB"})]
    assert majors_plan(weak, farm, 15).loc["P16", "Action"] == roster.PROMOTE  # a better bat replaces the weakest bat


def test_the_bat_only_hitters_of_the_majors_do_not_lean_to_one_side_and_a_switch_hitter_is_worth_a_little_more():
    def fielders(position, n):
        return [("MLB", False, {"Listed": position, "Value": 70.0, "Tier": "MLB"}) for _ in range(n)]

    def bat(value):
        return {"Listed": "DH", "Value": value, "Tier": "MLB"}

    base = fielders("C", 2) + fielders("SS", 5) + fielders("LF", 4)  # P1-P11
    mlb = [*base, ("MLB", False, bat(90.0)), ("MLB", False, bat(60.0))]  # P12, P13: two left-handed bats
    farm = [("AAA", False, {**bat(80.0)}), ("AAA", False, {**bat(79.0)})]  # P14 left-handed, P15 right-handed
    left, right = {12: 2, 13: 2, 14: 2, 15: 1}, {}
    plan = majors_plan(mlb, farm, 13, left)
    assert (
        plan.loc["P15", "Action"] == roster.PROMOTE and plan.loc["P14", "Action"] == ""
    )  # the right-hander balances the pair
    assert plan.loc["P13", "Action"] == roster.DEMOTE
    alike = majors_plan(mlb, farm, 13, {**left, 15: 2})
    assert alike.loc["P14", "Action"] == roster.PROMOTE  # two left-handers: the better one
    switch = majors_plan(mlb, [("AAA", False, bat(79.0)), ("AAA", False, bat(80.0))], 13, {12: 2, 13: 1, 14: 3, 15: 1})
    assert (
        switch.loc["P14", "Action"] == roster.PROMOTE and switch.loc["P15", "Action"] == ""
    )  # a right-hander a hair better: the switch hitter
    assert right == {}
