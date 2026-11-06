"""The roster plan of the level moves page: pure functions over the table of `minors.manage`, no UI.

`manage` says, player by player, who should move up or down by his ratings. The plan turns that into what to do with the whole
organization under the limits of every level (config/saves.toml): the age and pro-years limits, the most players a level can hold
and how many of them should be pitchers. Every player gets one destination (stay, the level above, the level below or out), chosen
by an integer program that follows the wishes of `manage` as far as the limits allow:

* a move is only to an adjacent level whose age and pro-years limits the player meets (a 26 year old does not go down to a level
  with a maximum age of 25); a player over the limits of his own level cannot stay;
* a level never holds more than its maximum (a violation is possible only when nothing else can be done, and it is reported);
* the pitchers of a level are kept within `pitcher_slack` of its target (`pitchers`, a target, not a maximum: pitchers are hurt
  more often, so the level should have that many) when the choice allows it: a level with too few gets pitchers from the levels
  next to it, one with too many sends its extra ones to a level that is short, and when a level sheds players the cut comes from
  the side that has too many. A move made only for the count costs what the player is worth keeping, so prospects are not
  moved for it;
* a promotion nobody asked for costs more than a demotion, so a level that is too full sends its fillers down, not up (and the
  top level never sends one to the majors to make room);
* a player is released (DFA when he is on the 40-man roster) only when no level can take him, and the one with the least reason to
  be kept goes first (the prospects with development left stay).

The plan also names the hitters listed at a position that is not their best and the pitchers whose role should change (a level
wants `starters` starting pitchers; a reliever who belongs in the rotation is flagged).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from util.config import LevelLimits
from viewer import depth, minors, rule5

PROMOTE, DEMOTE, DFA, RELEASE, DFA_OR_RELEASE = "Promote", "Demote", "DFA", "Release", "DFA / Release"
HITTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")
LINEUP = (*HITTER_POSITIONS, "DH")
INFIELD, OUTFIELD = ("1B", "2B", "3B", "SS"), ("LF", "CF", "RF")
COVER_INFIELD = (
    "2B",
    "3B",
    "SS",
)  # a first baseman covers first base only: he is not one of the infielders a team needs
CAPABLE_VALUE = (
    0.0  # hand tuned: runs of value at a position from which a player counts as able to cover it (catchers: any)
)
BAT_FIRST_GAP = (
    10.0  # hand tuned: runs by which a player's bat (1B or DH) must beat his best field position to make him a bat
)
MLB_PITCHER_SLACK = (
    0  # the majors carry exactly 13 pitchers and 13 hitters in the season (the minors may be one over or under)
)
SWITCH_BONUS = (
    1.0  # hand tuned: what a switch hitter is worth on top (utility points; 30 = the best major leaguer): he faces the
)
# weaker side of every pitcher, but how much depends on his weak side, which is not known here
SIDE_BALANCE = (
    1  # the bat-only hitters of the majors: at most this many more left-handed bats than right-handed ones, or reverse
)
SIDE_PENALTY = 15.0  # hand tuned: per bat beyond that
BAT_ONLY_MAX = 3  # hand tuned: hitters of the majors who cover no position (a first baseman and a DH platoon of two)
BAT_ONLY_PENALTY = 20.0  # per bat-only hitter beyond that: a full-time DH moves one of the platoon down
NEEDS = {"C": 2, "IF": 5, "OF": 4}  # what every level (the majors too) must have to field a team, by listed position
COVER_PENALTY = (
    40.0  # per player short of that: more than any move that is not wanted, less than breaking a roster maximum
)
CAPACITY_PENALTY = 60.0  # per player above the maximum of a level: only when nothing else can be done
PITCHER_PENALTY = (
    6.0  # per pitcher beyond the slack of the target of a level (or hitter above the rest of a full level)
)
NEAR_MLB_DEPTH = 4  # how many levels down from the top a player close to the majors still counts for something
MLB_UTILITY = (
    30.0  # what the majors are worth to a player: this times where his current value stands among the league's
)
# major leaguers of his kind (0 to 1), so the roster of the majors goes to the best players the organization has
MLB_FRICTION = 1.0  # a move to or from the majors that gains almost nothing is not made
BELOW_FLOOR_PENALTY = (
    10.0  # promoting a player below the floor of the majors for his value (not for what the roster needs)
)
MLB_RELEASE_EXTRA = 15.0  # releasing a major leaguer is a bigger step: only when the roster limit leaves no other way
RELEASE_BASE = 3.0  # a release costs more than the demotion of a filler, less than the demotion of a prospect
BOTTOM_RELEASE = 3.0  # hand tuned: a filler of the bottom level costs this + a quarter of keep + the quality term below
# QUALITY_RELEASE: hand tuned, what releasing the best player costs more than the worst (below the capacity penalty, so a level
# may rather stay over its maximum than lose its best pitcher). The quality is the larger of where his value now stands among the
# league's major leaguers and, for a prospect with development left, his ceiling.
QUALITY_RELEASE = 30.0


@dataclass(frozen=True)
class Plan:
    players: pd.DataFrame  # the table of manage with Action, Dest, Why, Final
    positions: pd.DataFrame  # the players listed at a position that is not their best
    summary: pd.DataFrame  # per level: players and pitchers now and after the plan, against the limits


def _weight(row: pd.Series) -> float:
    """How much it matters to move him as the ratings say: the prospects with development left more than the rest."""
    if row["Tier"] == minors.TOP_PROSPECT and row["focus"]:
        return 5.0
    return 3.0 if row["focus"] else 1.0


def _keep(row: pd.Series, depth: int, rules: minors.Rules) -> float:
    """What a player is worth keeping where he is (it prices a demotion or a release: the more, the later he goes). `depth` is
    the place of his level on the ladder, 0 = the top level.

    Two things make a minor leaguer worth keeping. A prospect with development left, by his ceiling: the lower the level, the
    more that counts (it is where prospects are made). A player who is close to the majors, by how his key ratings stand at the
    level above: that counts at the top level, where the roster is the last stop before the majors and the cover for injuries
    there, less at each level down, and nothing at the bottom, where a player without prospect status is only filler."""
    if row.get("Level") == minors.TOP:  # on the active roster: by how he stands among the league's major leaguers
        return 1.0 + 10.0 * (row["mean"] if pd.notna(row["mean"]) else 0.5)
    prospect = 0.0
    if row["Window"] > 0 and pd.notna(row["ceiling_pct"]):
        prospect = 8.0 * row["ceiling_pct"] * (1.0 + 0.25 * depth)
    near = 0.0
    if pd.notna(row["next_mean"]) and depth < NEAR_MLB_DEPTH:
        near = 6.0 * row["next_mean"] * (1.0 - depth / NEAR_MLB_DEPTH)
    return 1.0 + max(prospect, near, 3.0 * (row["mean"] if pd.notna(row["mean"]) else 0.5) * (depth == 0))


def bottom_level(rungs: list[str], levels: dict[str, LevelLimits]) -> str | None:
    """The lowest level of the ladder that players are demoted into: not the international complex (placed automatically) and
    not the winter league (off the ladder). Nobody can be demoted below it, so its extra players are released."""
    open_ = [lv for lv in rungs if lv != minors.TOP and not levels[lv].no_demotion]
    return open_[-1] if open_ else None


NO_FUTURE_CEILING = (
    0.03  # hand tuned: a player whose ceiling stands below this share of the league's major leaguers has no future
)
NO_FUTURE_GAIN = (
    4.0  # hand tuned: what releasing him is worth; the cover of a level (40) and its pitchers (6) are worth more
)
LEVEL_FLOOR_SHARE = 0.8  # hand tuned: releasing for no future never takes a level below this share of its maximum


def _quality(row: pd.Series) -> float:
    """0 to 1: how much a player is worth having, by what he is now (his standing among the league's major leaguers) or what
    he can become (the ceiling of a prospect with development left). A veteran past his window counts by his value now only.
    """
    now = row["mlb_pct"] if pd.notna(row.get("mlb_pct")) else 0.0
    later = row["ceiling_pct"] if row.get("Window", 0) > 0 and pd.notna(row.get("ceiling_pct")) else 0.0
    return float(max(now, later))


def _options(row: pd.Series, rungs: list[str], levels: dict[str, LevelLimits], rules: minors.Rules) -> list:
    """The (destination, utility) pairs open to a player: stay, up, down, out (None). The utility is how well the destination
    follows the move the ratings ask for; a move nobody asked for costs more the more the player is worth keeping, so the
    fillers are demoted before the prospects, and a release (cheaper for a filler than a demotion of a prospect) is the last
    resort. Who plays in the majors is decided before (`_solve_majors`): this is for the rest."""
    level, age, years = row["Level"], row["Age"], row["Pro"]
    i = rungs.index(level)
    offset = (
        1 if rungs[0] == minors.TOP else 0
    )  # the majors are a level of the plan, not a rung of the minor league ladder
    wish, weight, keep = row["Move"], _weight(row), _keep(row, i - offset, rules)
    wants_up = wish in (minors.UP, minors.MUST)
    options = []
    if not minors._fits(level, age, years, levels):
        options.append((level, 0.0 if not wish else (-weight if wants_up else -1.0)))
    above = rungs[i - 1] if i > 0 else (minors.TOP if level != minors.TOP else None)
    majors_in_plan = rungs[0] == minors.TOP  # then who goes to the majors is decided first (`_solve_majors`)
    if (
        above
        and (above != minors.TOP or (wants_up and not majors_in_plan))
        and not minors._fits(above, age, years, levels)
    ):
        options.append((above, weight if wants_up else -(2.0 + 1.2 * keep)))
    below = rungs[i + 1] if i + 1 < len(rungs) else None
    if below and levels[below].no_demotion:
        below = None  # the international complex: players are placed there automatically, nobody is demoted into it
    if below and not minors._fits(below, age, years, levels):
        options.append(
            (
                below,
                (1.0 if wish == minors.DOWN else -(1.0 + 1.2 * keep)),
            )
        )
    if row.get("hopeless"):  # no future in the organization: out, unless the level needs him to field a team
        options.append((None, NO_FUTURE_GAIN))
    elif (
        level == bottom_level(rungs, levels) and not row["focus"]
    ):  # a filler of the bottom level goes first: no level below
        options.append((None, -(BOTTOM_RELEASE + 0.25 * keep + QUALITY_RELEASE * _quality(row))))
    else:
        extra = MLB_RELEASE_EXTRA if level == minors.TOP else QUALITY_RELEASE * _quality(row)
        options.append((None, -(RELEASE_BASE + 1.5 * keep + extra)))
    return options


def capability(table: pd.DataFrame, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Which players of the table can cover a group of positions, by what they can defend and not by the position the game lists
    (a left fielder who cannot field is a DH; a player whose bat beats his best position by `BAT_FIRST_GAP` runs is a bat first
    and covers nothing, whatever else he can play): C (he has a value as a catcher; catchers are all below the others), IF
    (value of at least `CAPABLE_VALUE` at 2B, 3B or SS) and OF (at LF, CF or RF), several at once for a utility man, and BAT, a
    hitter who covers none of them (first basemen and designated hitters, who play by their bat). A player the outlook has no
    position value for is placed by his listed position."""
    values = frame.loc[frame["slot"].isin(HITTER_POSITIONS), ["ID", "slot", "value_now"]]
    able = {
        "C": set(values.loc[values["slot"] == "C", "ID"]),
        "IF": set(values.loc[values["slot"].isin(COVER_INFIELD) & (values["value_now"] >= CAPABLE_VALUE), "ID"]),
        "OF": set(values.loc[values["slot"].isin(OUTFIELD) & (values["value_now"] >= CAPABLE_VALUE), "ID"]),
    }
    best_field = values[values["slot"].isin([*COVER_INFIELD, *OUTFIELD])].groupby("ID")["value_now"].max()
    best_bat = values[values["slot"].isin(["1B", "DH"])].groupby("ID")["value_now"].max()
    bat_first = set(best_bat.index[(best_bat - best_field.reindex(best_bat.index)) > BAT_FIRST_GAP])
    for group in ("IF", "OF"):
        able[group] -= bat_first  # a first baseman who could also play right field covers right field in no real sense
    known = set(frame["ID"])
    listed = table["Listed"].map(_role) if "Listed" in table else pd.Series("", index=table.index)
    out = {
        group: np.array([(i in able[group]) if i in known else role == group for i, role in zip(table["ID"], listed)])
        for group in able
    }
    hitter = ~table["pitcher"].to_numpy(dtype=bool)
    out = {group: able_ & hitter for group, able_ in out.items()}
    out["BAT"] = hitter & ~(out["C"] | out["IF"] | out["OF"])
    return out


def batting_sides(table: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """+1 for a left-handed bat, -1 for a right-handed one, 0 for a switch hitter (or when the export has no `B`): the bat-only
    hitters of the majors should not lean to one side, since a platoon at DH needs one of each."""
    if "B" not in frame:
        return np.zeros(len(table))
    side = frame.drop_duplicates("ID").set_index("ID")["B"]
    return table["ID"].map(side).map({1: -1.0, 2: 1.0, 3: 0.0}).fillna(0.0).to_numpy()


def switch_hitters(table: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    if "B" not in frame:
        return np.zeros(len(table), dtype=bool)
    return table["ID"].map(frame.drop_duplicates("ID").set_index("ID")["B"]).eq(3).to_numpy()


def held(table: pd.DataFrame) -> np.ndarray:
    """The players who are injured now (`INJ` of the export is Yes): they are not moved, up or down, until an export says they are
    well again, and they do not count as cover for a position. Without the column nobody is held."""
    if "INJ" not in table:
        return np.zeros(len(table), dtype=bool)
    return rule5.flagged(table["INJ"]).to_numpy(dtype=bool)


def injury_text(row: pd.Series) -> str:
    left = str(row.get("Left", "")).strip()
    return f"injured ({left if left not in ('', 'nan', '-') else 'time left unknown'}): not moved until he is well"


def _number(value) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else 0.0


def can_option(row: pd.Series, rules: minors.Rules) -> bool:
    """Whether a major leaguer can be sent down without waivers. A player has `max_option_years` (3) option years; the first time
    he is sent down in a season uses one (`OPT` counts them), and once he has been optioned that season (`OY` >= 1) he can go
    down and up as often as the season lasts. So: optioned this season already, or option years left (`OPT` below the maximum).
    Without the `OPT` column (the export of the league has none) there is no information and he can."""
    if "OPT" not in row.index or pd.isna(row["OPT"]):
        return True
    return _number(row.get("OY")) >= 1 or _number(row["OPT"]) < rules.max_option_years


def option_text(row: pd.Series, rules: minors.Rules) -> str:
    """What the options of a major leaguer say, for the reason of a move (nothing when the export has no `OPT`)."""
    if "OPT" not in row.index or pd.isna(row["OPT"]):
        return ""
    used, this_year = int(_number(row["OPT"])), int(_number(row.get("OY")))
    if this_year >= 1:
        return f"already optioned this season (option year {used} of {rules.max_option_years})"
    if used < rules.max_option_years:
        return f"uses option year {used + 1} of {rules.max_option_years}"
    return f"no option left ({used} of {rules.max_option_years} used): waivers"


def _solve_majors(
    table: pd.DataFrame,
    rungs: list[str],
    levels: dict[str, LevelLimits],
    rules: minors.Rules,
    can_start: np.ndarray,
    injured: np.ndarray | None = None,
    covers: dict[str, np.ndarray] | None = None,
    sides: np.ndarray | None = None,
    switch: np.ndarray | None = None,
) -> dict[int, str | None]:
    """Who plays in the majors: the active roster goes to the best players the organization has. The candidates are the active
    roster and the players of the level below; a player is worth `MLB_UTILITY` times where his current value stands among the
    league's major leaguers of his kind. A major leaguer goes down only if he has an option left (`can_option`; else he is DFA'd,
    that is, exposed to waivers), the roster limit of the phase is kept, every group a team needs is covered (catchers,
    infielders, outfielders, starters) and in the season the pitchers are 13 within the slack. The rest of the organization is
    planned after this and adapts: a problem of AAA is solved from below, not by sending a major leaguer down.
    Returns row -> destination (the top level, the level below or None for out) for the rows of those two levels."""
    top, farm = rungs[0], (rungs[1] if len(rungs) > 1 else None)
    limits = levels[top]
    injured = np.zeros(len(table), dtype=bool) if injured is None else injured
    sides = np.zeros(len(table)) if sides is None else sides
    bonus = SWITCH_BONUS * (
        np.zeros(len(table)) if switch is None else switch
    )  # a switch hitter is worth a little more
    entries, sizes = [], {}
    for i, row in table.iterrows():
        pct = row["mlb_pct"] if pd.notna(row["mlb_pct"]) else 0.0
        if injured[i] and row["Level"] in (top, farm):  # stays where he is
            entries.append((i, row["Level"], MLB_UTILITY * pct + bonus[i] if row["Level"] == top else 0.0))
        elif row["Level"] == top:
            entries.append((i, top, MLB_UTILITY * pct + bonus[i]))
            if farm and can_option(row, rules) and not minors._fits(farm, row["Age"], row["Pro"], levels):
                entries.append((i, farm, -MLB_FRICTION))
            entries.append((i, None, -(RELEASE_BASE + 1.5 * _keep(row, 0, rules) + MLB_RELEASE_EXTRA)))
        elif row["Level"] == farm and not minors._fits(top, row["Age"], row["Pro"], levels):
            floor = BELOW_FLOOR_PENALTY if pct < rules.mlb_floor else 0.0
            entries.append((i, farm, 0.0))
            entries.append((i, top, MLB_UTILITY * pct + bonus[i] - MLB_FRICTION - floor))
        else:
            continue
        sizes[i] = None
    if not entries:
        return {}
    groups = list(_needs(top, rules))
    n = len(entries)
    extra = (
        1 + len(groups) + 3 + 1 + 2
    )  # roster limit, groups, pitchers, hitters over, bat-only over, left / right lean
    cost = np.zeros(n + extra)
    cost[:n] = [-u for _, _, u in entries]
    cost[n] = CAPACITY_PENALTY
    first_group, after_groups = n + 1, n + 1 + len(groups)
    cost[first_group:after_groups] = COVER_PENALTY
    cost[after_groups:] = PITCHER_PENALTY
    cost[n + 1 + len(groups) + 3] = BAT_ONLY_PENALTY
    cost[-2:] = SIDE_PENALTY
    ids = list(sizes)
    roles = table["Listed"].map(_role) if "Listed" in table else pd.Series("", index=table.index)
    pitcher = table["pitcher"]
    rows, lower, upper = [], [], []

    def add(members, slack_index, sign, low, high):
        line = np.zeros(n + extra)
        line[members] = 1
        if slack_index is not None:
            line[n + slack_index] = sign
        rows.append(line)
        lower.append(low)
        upper.append(high)

    for i in ids:
        add([k for k, (q, _, _) in enumerate(entries) if q == i], None, 0, 1, 1)
    at_top = [k for k, (_, dest, _) in enumerate(entries) if dest == top]
    add(at_top, 0, -1, -np.inf, limits.max_players)
    for c, group in enumerate(groups):
        members = [
            k for k, (i, dest, _) in enumerate(entries)
            if dest == top and not injured[i]
            and (can_start[i] if group == "SP" else covers[group][i] if covers else roles[i] == group)
        ]  # fmt: skip
        add(members, 1 + c, 1, _needs(top, rules)[group], np.inf)
    if covers:  # the hitters who play by their bat only: a first baseman and a DH platoon, not more
        bats = [k for k, (i, dest, _) in enumerate(entries) if dest == top and covers["BAT"][i]]
        add(bats, 1 + len(groups) + 3, -1, -np.inf, BAT_ONLY_MAX)
        lean = np.zeros(n + extra)  # left minus right bats among them (a switch hitter leans neither way)
        for k in bats:
            lean[k] = sides[entries[k][0]]
        for sign, slack in ((1, extra - 2), (-1, extra - 1)):
            line = sign * lean
            line[n + slack] = -1
            rows.append(line)
            lower.append(-np.inf)
            upper.append(SIDE_BALANCE)
    if limits.pitchers > 0:
        base = 1 + len(groups)
        mine = [k for k in at_top if pitcher[entries[k][0]]]
        add(mine, base, -1, -np.inf, limits.pitchers + MLB_PITCHER_SLACK)
        add(mine, base + 1, 1, limits.pitchers - MLB_PITCHER_SLACK, np.inf)
        hitters = [k for k in at_top if not pitcher[entries[k][0]]]
        add(hitters, base + 2, -1, -np.inf, limits.max_players - limits.pitchers + MLB_PITCHER_SLACK)
    result = milp(
        cost,
        constraints=LinearConstraint(np.array(rows), lower, upper),
        integrality=np.r_[np.ones(n), np.zeros(extra)],
        bounds=Bounds(0, np.r_[np.ones(n), np.full(extra, np.inf)]),
    )
    if not result.success:
        return {}
    return {i: dest for k, (i, dest, _) in enumerate(entries) if result.x[k] > 0.5}


def _role(listed) -> str:
    """The group a player counts in when a level needs a team: C, IF (2B, 3B, SS: a first baseman covers first base only), OF
    (LF, CF, RF) or "" (1B, DH, pitchers)."""
    return "C" if listed == "C" else "IF" if listed in COVER_INFIELD else "OF" if listed in OUTFIELD else ""


def _needs(level: str, rules: minors.Rules) -> dict[str, int]:
    """What a level needs: two catchers, five infielders, four outfielders and the starters of its rotation (5 in the majors,
    `starters` below) who can start (the listed starters and the relievers ranked as starters)."""
    return {**NEEDS, "SP": _starters(level, rules)}


def _fixed_options(row: pd.Series, dest, rungs: list[str], levels: dict[str, LevelLimits], rules: minors.Rules) -> list:
    """The options of a player the majors decided on. A player who stays in the majors or is promoted or is DFA'd has none. A
    major leaguer who is sent down can go to any level of the minors whose limits he meets (an optioned player is not tied to
    the level below): the nearer to the majors the better, by what he is worth keeping, so he takes a place from the fillers of a
    full AAA rather than the other way around."""
    if dest in (None, minors.TOP) or row["Level"] != minors.TOP:
        return [(dest, 0.0)]
    keep = _keep(row, 0, rules)
    ladder = rungs[1:]
    options = [
        (level, -step * (1.0 + 1.2 * keep))
        for step, level in enumerate(ladder)
        if not levels[level].no_demotion and not minors._fits(level, row["Age"], row["Pro"], levels)
    ]
    return options or [(dest, 0.0)]


def _solve(
    table: pd.DataFrame,
    rungs: list[str],
    levels: dict[str, LevelLimits],
    rules: minors.Rules,
    can_start: np.ndarray,
    fixed: dict[int, str | None] | None = None,
    injured: np.ndarray | None = None,
    playing: dict | None = None,
) -> list[str | None]:
    """The destination of every row of `table` (a level name, the top level above the ladder or None for out). With `playing`
    (`lineup.joint_values`) the plan also decides who plays: see `_with_playing`."""
    fixed = fixed or {}  # decided before: the majors and the players promoted to them
    choices = [
        [(fixed[p], 0.0)] if p in fixed else _options(row, rungs, levels, rules)
        for p, (_, row) in enumerate(table.iterrows())
    ]
    slack = rules.pitcher_slack
    pairs = [(p, dest, utility) for p, options in enumerate(choices) for dest, utility in options]
    capped = [lv for lv in rungs if levels[lv].max_players > 0]
    targeted = [lv for lv in rungs if levels[lv].pitchers > 0]
    staffed = [  # the levels that have a team now; the complex where players are placed has no team to field
        lv for lv in rungs if (table["Level"] == lv).any() and not levels[lv].no_demotion and "Listed" in table
    ]
    groups = ("C", "IF", "OF", "SP")
    n, extra = len(pairs), len(capped) + 3 * len(targeted) + len(groups) * len(staffed)
    cost = np.zeros(n + extra)
    cost[:n] = [-u for _, _, u in pairs]
    first_pitcher = n + len(capped)
    first_cover = first_pitcher + 3 * len(targeted)
    cost[n:first_pitcher] = CAPACITY_PENALTY
    cost[first_pitcher:first_cover] = PITCHER_PENALTY
    cost[first_cover:] = COVER_PENALTY
    roles = table["Listed"].map(_role).to_numpy() if "Listed" in table else np.full(len(table), "")
    if injured is not None:
        roles, can_start = np.where(injured, "", roles), can_start & ~injured  # an injured player covers nothing
    pitcher = table["pitcher"].to_numpy(dtype=bool)
    rows, lower, upper = [], [], []

    def add(coefficients: np.ndarray, low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(low)
        upper.append(high)

    for p in range(len(table)):  # one destination each
        line = np.zeros(n + extra)
        line[[k for k, (q, _, _) in enumerate(pairs) if q == p]] = 1
        add(line, 1, 1)
    for j, level in enumerate(capped):  # at most the maximum, unless there is no other way
        line = np.zeros(n + extra)
        line[[k for k, (_, dest, _) in enumerate(pairs) if dest == level]] = 1
        line[n + j] = -1
        add(line, -np.inf, levels[level].max_players)
    for j, level in enumerate(
        targeted
    ):  # the pitchers within the slack of the target, the hitters of a full level within the slack of the rest, softly
        limits = levels[level]
        for k, (kind, share) in enumerate(
            (("pitchers", limits.pitchers), ("hitters", limits.max_players - limits.pitchers))
        ):
            if k == 1 and limits.max_players <= 0:
                continue  # no maximum: no split to keep
            mine = [k2 for k2, (p, dest, _) in enumerate(pairs) if dest == level and pitcher[p] == (kind == "pitchers")]
            line = np.zeros(n + extra)
            line[mine] = 1
            line[first_pitcher + 3 * j + k] = -1
            add(line, -np.inf, share + slack)
        short = np.zeros(n + extra)  # too few pitchers: injuries are more frequent among them, so the target is kept
        short[[k2 for k2, (p, dest, _) in enumerate(pairs) if dest == level and pitcher[p]]] = 1
        short[first_pitcher + 3 * j + 2] = 1
        add(short, limits.pitchers - slack, np.inf)
    for j, level in enumerate(staffed):  # a team that can play: catchers, infielders, outfielders, a rotation
        for c, (group, need) in enumerate(_needs(level, rules).items()):
            mine = [
                k for k, (p, dest, _) in enumerate(pairs)
                if dest == level and (can_start[p] if group == "SP" else roles[p] == group)
            ]  # fmt: skip
            line = np.zeros(n + extra)
            line[mine] = 1
            line[first_cover + len(groups) * j + c] = 1
            add(line, need, np.inf)
    matrix = sparse.csr_matrix(np.array(rows))
    integrality, top = np.r_[np.ones(n), np.zeros(extra)], np.r_[np.ones(n), np.full(extra, np.inf)]
    if playing:
        matrix, cost, lower, upper, integrality, top = _with_playing(
            matrix, cost, lower, upper, integrality, top, pairs, playing
        )
    result = milp(
        cost,
        constraints=LinearConstraint(matrix, lower, upper),
        integrality=integrality,
        bounds=Bounds(0, top),
    )
    if not result.success:  # cannot happen (the slack makes it feasible); fall back to the ratings' wishes
        return [row["To"] if row["Move"] and not row["Blocked"] else row["Level"] for _, row in table.iterrows()]
    chosen = result.x[:n] > 0.5
    dest = [None] * len(table)
    for k, (p, level, _) in enumerate(pairs):
        if chosen[k]:
            dest[p] = level
    return dest


def _with_playing(matrix, cost, lower, upper, integrality, top, pairs, playing: dict):
    """The program of `_solve` with the slots of the levels: the value of who plays where is part of the choice of the level.

    `playing` has `weight`, `base` ((row, level) -> what he is worth on the bench or in the bullpen there: it joins the value of
    the destination), `inc` (a table with row, level, slot and the gain of a slot over the bench) and `caps` ((level, slot) -> how
    many can play it: one for a lineup slot, the starters wanted for the rotation). A slot variable can be 1 only when its player
    goes to its level (y <= x) and a slot holds at most its cap; the objective gains `weight` x the value of every slot filled.
    """
    weight = playing["weight"]
    index = {(p, dest): k for k, (p, dest, _) in enumerate(pairs)}
    cost = cost.copy()
    for (p, level), value in playing["base"].items():
        if (p, level) in index:
            cost[index[(p, level)]] -= weight * value
    inc = playing["inc"]
    inc = inc[[(p, level) in index for p, level in zip(inc["row"], inc["level"])]].reset_index(drop=True)
    m, width = len(inc), matrix.shape[1]
    if m == 0:
        return matrix, cost, lower, upper, integrality, top
    new_rows, new_cols, new_vals, new_low, new_high = [], [], [], [], []
    for j, (p, level) in enumerate(zip(inc["row"], inc["level"])):  # y <= x
        new_rows += [j, j]
        new_cols += [width + j, index[(p, level)]]
        new_vals += [1.0, -1.0]
        new_low.append(-np.inf)
        new_high.append(0.0)
    r = m
    for (level, slot), group in inc.groupby(["level", "slot"]):  # a slot holds at most its cap
        new_rows += [r] * len(group)
        new_cols += [width + j for j in group.index]
        new_vals += [1.0] * len(group)
        new_low.append(-np.inf)
        new_high.append(playing["caps"].get((level, slot), 1))
        r += 1
    extra_rows = sparse.csr_matrix((new_vals, (new_rows, new_cols)), shape=(r, width + m))
    matrix = sparse.vstack([sparse.hstack([matrix, sparse.csr_matrix((matrix.shape[0], m))]), extra_rows], format="csr")
    cost = np.r_[cost, -weight * inc["gain"].to_numpy(dtype=float)]
    return (
        matrix, cost, np.r_[lower, new_low], np.r_[upper, new_high], np.r_[integrality, np.ones(m)], np.r_[top, np.ones(m)],
    )  # fmt: skip


def _why(
    row: pd.Series, dest, counts: pd.Series, levels: dict[str, LevelLimits], rungs: list[str], rules: minors.Rules
) -> str:
    level = row["Level"]
    limit = levels[level].max_players
    full = f"{level} holds at most {limit} players" if limit and counts.get(level, 0) > limit else ""
    asked = row["Move"] and not row["Blocked"] and dest == row["To"]
    if dest == level:
        return ""
    if level == minors.TOP:
        note = option_text(row, rules)
        if dest is None:
            return (
                f"a spot is needed (the active roster is {limit}) and he cannot be sent down: {note or 'no room below'}"
            )
        return "; ".join(
            part for part in (f"a better player takes the spot, or the roster limit of {limit}", note) if part
        )
    if dest == minors.TOP:
        return f"among the best the majors can use: his value stands at {row['mlb_pct']:.0%} of the league's major leaguers"
    if dest is None and row.get("hopeless"):
        return f"no future: his ceiling stands at {row['ceil_mlb']:.0%} of the league's major leaguers"
    if dest is None:
        over = minors._fits(level, row["Age"], row["Pro"], levels)
        if over:
            return f"over the limit of the level ({over}) and no level takes him"
        return (full or "roster spot") + " and no level below takes him; the least reason to keep"
    if asked:
        return row["Reason"]
    return f"to make room: {full}" if full else "roster balance (pitchers / hitters / the players a level needs)"


def mlb_peers(frame: pd.DataFrame, ratings: pd.DataFrame, rules: minors.Rules) -> pd.DataFrame:
    """The league's major leaguers (every organization): pitcher flag and current value, what the players of the organization
    are measured against for the majors."""
    players = minors.league_players(frame, ratings, rules)
    return players.loc[players["team_level"] == minors.TOP, ["pitcher", "value_now"]]


def mlb_table(frame: pd.DataFrame, ratings: pd.DataFrame, org: str, rules: minors.Rules, version: str) -> pd.DataFrame:
    """The active roster of `org` in the majors (the `ACT` flag of the export) as rows for `plan`: like the table of
    `minors.manage`, with Level "MLB" and no wish to move (`plan` sets in `mean` where each stands among the league's major
    leaguers of his kind, which is what keeps him)."""
    players = minors.league_players(frame, ratings, rules)
    players = players[players["team_level"] == minors.TOP]
    if "ACT" not in players or players.empty:
        return pd.DataFrame()
    mine = players[(players["org"] == str(org)) & rule5.flagged(players["ACT"])].copy()
    if mine.empty:
        return pd.DataFrame()
    mine = minors.add_tier(mine, rules, version)
    mine["Tier"], mine["focus"] = minors.TOP, False
    mine["Player"] = (mine["First Name"].astype(str) + " " + mine["Last Name"].astype(str)).map(depth._tidy)
    mine = mine.rename(
        columns={"slot": "Pos", "PROY": "Pro", "team_level": "Level", "value_now": "Value", "LPOS": "Listed"}
    )
    for column in ("Move", "To", "Reason", "Blocked"):
        mine[column] = ""
    mine["green"], mine["next_mean"], mine["mean"] = 0, np.nan, np.nan
    return mine.drop(columns=["First Name", "Last Name", "org"]).reset_index(drop=True)


def detect_limit(frame: pd.DataFrame, rules: minors.Rules) -> int | None:
    """The active roster limit the league is at, from how many players the typical team has on its active roster: the nearest
    of the season, offseason and spring training limits (None when the export has no `ACT` flag for the majors)."""
    if "ACT" not in frame:
        return None
    active = frame[(frame["team_level"] == minors.TOP) & rule5.flagged(frame["ACT"])].drop_duplicates("ID")
    if active.empty:
        return None
    typical = active.groupby("org").size().median()
    return min((rules.mlb_season, rules.mlb_offseason, rules.mlb_spring), key=lambda limit: abs(limit - typical))


def mlb_limits(rules: minors.Rules, limit: int) -> LevelLimits:
    """The majors as a level: the active roster `limit` of the phase (26, 40 or 60) and, in the season, the pitchers wanted."""
    return LevelLimits(max_players=limit, pitchers=rules.mlb_pitchers if limit == rules.mlb_season else 0)


def _starters(level: str, rules: minors.Rules) -> int:
    return rules.mlb_starters if level == minors.TOP else rules.starters


def _mlb_standing(table: pd.DataFrame, peers: pd.DataFrame | None, column: str = "Value") -> pd.Series:
    """Where each player's current value (or, with `column`, his ceiling) stands among the league's major leaguers of his kind
    (0 to 1); NaN without peers."""
    out = pd.Series(np.nan, index=table.index)
    if peers is None or column not in table:
        return out
    for pitcher in (False, True):
        mine = table["pitcher"] == pitcher
        here = peers.loc[peers["pitcher"] == pitcher, "value_now"].to_numpy(dtype=float)
        if mine.any() and len(here):
            out[mine] = minors.percentile(table.loc[mine, column], here)
    return out


def no_future(table: pd.DataFrame, levels: dict[str, LevelLimits], rungs: list[str]) -> pd.Series:
    """The players the organization has no reason to keep: a ceiling (`Ceil`) below `NO_FUTURE_CEILING` of the league's major
    leaguers of his kind. Per level the worst ceilings first, and never more than takes the level down to `LEVEL_FLOOR_SHARE`
    of its maximum (a level needs players to field a team). Not the majors, not the international complex."""
    out = pd.Series(False, index=table.index)
    for level in rungs:
        limits = levels[level]
        if level == minors.TOP or limits.no_demotion or limits.max_players <= 0:
            continue
        here = table[(table["Level"] == level) & (table["ceil_mlb"] < NO_FUTURE_CEILING)].sort_values("ceil_mlb")
        room = int((table["Level"] == level).sum() - np.ceil(LEVEL_FLOOR_SHARE * limits.max_players))
        out[here.index[: max(room, 0)]] = True
    return out


def plan(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    levels: dict[str, LevelLimits],
    rules: minors.Rules,
    mlb: pd.DataFrame | None = None,
    mlb_limit: int = 0,
    peers: pd.DataFrame | None = None,
    playing=None,
) -> Plan:
    """The plan for the table of `minors.manage` (every minor leaguer of one organization on the ladder). `frame` is the
    outlook frame, for the values at the positions. With `mlb` (the active roster, `mlb_table`) the majors are the top level
    of the plan with the roster limit `mlb_limit`, filled with the best players of the organization (`peers` = `mlb_peers`:
    the league's major leaguers they are measured against); players without an option left cannot be sent down (they stay or
    are DFA'd), and what the rest of the organization needs is solved below the majors. `playing` (`lineup.joint`) is a function
    of the table and the rungs that gives the slots of the levels: with it the choice of the level also follows who plays where.
    """
    rungs = minors.ladder(levels)
    if mlb is not None and not mlb.empty:
        levels = {minors.TOP: mlb_limits(rules, mlb_limit), **levels}
        rungs = [minors.TOP, *rungs]
        table = pd.concat([mlb, table], ignore_index=True)
    table = table.reset_index(drop=True).copy()
    table["mlb_pct"] = _mlb_standing(table, peers)
    table["ceil_mlb"] = _mlb_standing(table, peers, "Ceil")
    table["hopeless"] = no_future(table, levels, rungs) if peers is not None else False
    if rungs[0] == minors.TOP:
        table["mean"] = table["mean"].where(table["Level"] != minors.TOP, table["mlb_pct"])
    starters = set(frame.loc[frame["slot"] == "SP", "ID"])
    can_start = (
        table["pitcher"] & (table["ID"].isin(starters) | (table["Listed"] == "SP" if "Listed" in table else False))
    ).to_numpy()
    injured = held(table)
    covers = capability(table, frame) if rungs[0] == minors.TOP else None
    sides, switch = batting_sides(table, frame), switch_hitters(table, frame)
    chosen = (
        _solve_majors(table, rungs, levels, rules, can_start, injured, covers, sides, switch)
        if rungs[0] == minors.TOP
        else {}
    )
    fixed = {i: d for i, d in chosen.items() if table.at[i, "Level"] == minors.TOP or d == minors.TOP}
    fixed.update({i: table.at[i, "Level"] for i in np.flatnonzero(injured)})  # an injured player stays where he is
    dest = _solve(
        table, rungs, levels, rules, can_start, fixed, injured, playing(table, rungs, levels) if playing else None
    )
    table["can_start"] = can_start
    arriving = pd.Series([d for d, level in zip(dest, table["Level"]) if d is not None and d != level]).value_counts()
    counts = table["Level"].value_counts().add(arriving, fill_value=0)  # the players there and the ones sent to it
    on_forty = rule5.flagged(table["ON40"]) if "ON40" in table else pd.Series(False, index=table.index)
    exact = "ON40" in table
    table["Dest"] = [d if d is not None else "" for d in dest]
    table["Final"] = [d for d in dest]
    action = []
    for i, d in enumerate(dest):
        level = table.at[i, "Level"]
        if d is None:
            action.append(DFA if on_forty[i] else RELEASE if exact else DFA_OR_RELEASE)
        elif d == level:
            action.append("")
        else:
            action.append(PROMOTE if d == minors.TOP or rungs.index(d) < rungs.index(level) else DEMOTE)
    table["Action"] = action
    table["Why"] = [_why(table.loc[i], d, counts, levels, rungs, rules) for i, d in enumerate(dest)]
    for i in np.flatnonzero(injured):  # say why a player the ratings would move stays
        row = table.loc[i]
        if row["Move"] or (row["Level"] != minors.TOP and row["mlb_pct"] >= rules.mlb_floor):
            table.at[i, "Why"] = injury_text(row)
    positions = listed_positions(table, frame, rules)
    return Plan(table, positions, summary(table, positions, levels, rungs, rules))


def listed_positions(table: pd.DataFrame, frame: pd.DataFrame, rules: minors.Rules) -> pd.DataFrame:
    """The players whose listed position is not their best: a hitter who would add `position_gap` runs or more at another
    position in the best lineup of his level (the way the depth chart places the hitters: one per position, the most total value;
    or he is not ranked at the position he is listed at), a reliever who belongs in the rotation (the `starters` best by the
    value as a starter among the pitchers of his level after the plan) and a starter who does not. Columns: ID, Level, Player,
    Age, Listed, Better, Gain (runs; NaN when he is not ranked at his listed position), Why, At (his level after the plan).
    """
    if "Listed" not in table:
        return pd.DataFrame(columns=["ID", "Level", "Player", "Age", "Listed", "Better", "Gain", "Why", "At"])
    values = (
        frame[frame["ID"].isin(table["ID"])]
        .drop_duplicates(["ID", "slot"])
        .pivot(index="ID", columns="slot", values="value_now")
    )
    out = []
    lineup = dict.fromkeys(LINEUP, 1)
    for level, group in table[~table["pitcher"] & table["Final"].isin(table["Level"].unique())].groupby("Final"):
        pool = frame[frame["ID"].isin(group["ID"]) & frame["slot"].isin(lineup)]
        placed = {i: slot for slot, ids in depth.fill(pool, "value_now", lineup, 1).items() for i in ids}
        for (
            _,
            row,
        ) in group.iterrows():  # the best lineup of the level's hitters; the bench has no position to get wrong
            best = placed.get(row["ID"])
            if best is None or best == "DH" or row["Listed"] not in HITTER_POSITIONS or best == row["Listed"]:
                continue
            mine = values.loc[row["ID"]] if row["ID"] in values.index else pd.Series(dtype=float)
            listed = mine.get(row["Listed"], np.nan)
            gain = mine.get(best, np.nan) - listed if pd.notna(listed) else np.nan
            if pd.isna(gain) or gain >= rules.position_gap:
                why = (
                    f"not ranked at {row['Listed']}; the best lineup of {level} plays him at {best}"
                    if pd.isna(gain)
                    else (f"the best lineup of {level} plays him at {best}: +{gain:.1f} runs over {row['Listed']}")
                )
                out.append((row["ID"], row["Level"], row["Player"], row["Age"], row["Listed"], best, gain, why, level))
    starters = values["SP"] if "SP" in values.columns else pd.Series(dtype=float)
    for level, group in table[table["pitcher"] & table["Final"].isin(table["Level"].unique())].groupby("Final"):
        ranked = group.assign(sp=group["ID"].map(starters)).dropna(subset=["sp"]).sort_values("sp", ascending=False)
        wanted = _starters(level, rules)
        rotation = set(ranked["ID"].head(wanted))
        for _, row in group.iterrows():
            if row["Listed"] in ("RP", "CL") and row["ID"] in rotation:
                out.append((row["ID"], row["Level"], row["Player"], row["Age"], row["Listed"], "SP", np.nan,
                            f"one of the {wanted} best starters at {level}", level))  # fmt: skip
            elif (
                row["Listed"] == "SP" and row["ID"] not in rotation and len(rotation) >= wanted and level != minors.TOP
            ):
                out.append((row["ID"], row["Level"], row["Player"], row["Age"], row["Listed"], "RP", np.nan,
                            f"not among the {wanted} best starters at {level}", level))  # fmt: skip
    columns = ["ID", "Level", "Player", "Age", "Listed", "Better", "Gain", "Why", "At"]
    return pd.DataFrame(out, columns=columns)


def _covered(after: pd.DataFrame) -> dict[str, int]:
    """What a level has after the plan of each group it needs: catchers, infielders, outfielders, pitchers who can start."""
    if "Listed" not in after:
        return {}
    roles = after["Listed"].map(_role)
    return {
        "Catchers": int((roles == "C").sum()), "Infielders": int((roles == "IF").sum()),
        "Outfielders": int((roles == "OF").sum()),
        "Can start": int(after["can_start"].sum()) if "can_start" in after else 0,
    }  # fmt: skip


def summary(
    table: pd.DataFrame, positions: pd.DataFrame, levels: dict[str, LevelLimits], rungs: list[str], rules: minors.Rules
) -> pd.DataFrame:
    """Per level of the ladder: the players and pitchers now and after the plan against the maximum and the pitcher target, the
    starters listed and the starters after the flagged role changes (the level wants `starters`)."""
    rows = []
    for level in rungs:
        now = table[table["Level"] == level]
        after = table[table["Final"] == level]
        if now.empty and after.empty:
            continue
        listed = int((after["Listed"] == "SP").sum()) if "Listed" in table else 0
        flagged = positions[positions["At"] == level] if len(positions) else positions  # at the level after the plan
        to_sp = int((flagged["Better"] == "SP").sum()) if len(flagged) else 0
        to_rp = int((flagged["Better"] == "RP").sum()) if len(flagged) else 0
        rows.append(
            {
                "Level": level, "Now": len(now), "After": len(after), "Max": levels[level].max_players,
                "Pitchers now": int(now["pitcher"].sum()), "Pitchers after": int(after["pitcher"].sum()),
                "Target": levels[level].pitchers, "Listed SP": listed, "SP after": listed + to_sp - to_rp,
                "Wanted SP": _starters(level, rules),
                **_covered(after),
            }
        )  # fmt: skip
    return pd.DataFrame(rows).set_index("Level") if rows else pd.DataFrame()
