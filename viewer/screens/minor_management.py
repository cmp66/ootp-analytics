"""The minor league management page: the moves of an organization by level (promote, demote, DFA or release, wrong position).

Follows how the game's ratings relative to a level are read: a key rating that is green among the players of the level (all
organizations) is a player who dominates it. See config/minors.toml for the rules and viewer/minors.py for the logic; the age and
pro-years limits of the levels are in config/saves.toml.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from util.update_config import load_update_config  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import served_seasons  # noqa: E402
from model import scales  # noqa: E402
from viewer import data_cache, league, lineup, minors, overview, playing, roster  # noqa: E402
from viewer.screens import org_view  # noqa: E402

ROW_HEIGHT = 35
WIDTHS = {
    "Player": 150, "Pos": 50, "Age": 50, "Pro": 50, "Tier": 105, "Win": 50, "Ceil": 56, "Floor": 56, "Top-50 %": 70, "Green": 56,
    "Mean %": 66, "Opt": 70, "To": 60, "Reason": 520, "Listed": 60, "Better": 60, "Gain": 56, "Slot": 60, "Primary": 64,
    "Value": 60, "Dev": 52, "Learn": 56, "Hit": 48, "Now": 52, "Grow": 56,
}  # fmt: skip
MOVE_COLORS = {minors.UP: "#2e9e57", minors.DOWN: "#d6453d", minors.MUST: "#e08a1e"}
TIER_COLORS = {minors.TOP_PROSPECT: "#2e9e57", minors.PROSPECT: "#5aa469"}
TEXT_COLUMNS = ("Player", "Pos", "Tier", "To", "Reason", "Listed", "Better", "Green", "Opt", "Slot", "Primary")


def load(save: str, season: int):
    return data_cache.outlook(save, season), data_cache.player_data(save, season), data_cache.growth(save, season)


def rules_from_page(base: minors.Rules) -> minors.Rules:
    """The rules of config/minors.toml, with the sliders of the page on top (for this session only)."""
    with st.expander("Rules (defaults from config/minors.toml; changes here last for this session)"):
        first, second, third = st.columns(3)
        green = first.slider(
            "Green from percentile",
            0.5,
            0.95,
            base.green,
            0.05,
            help="A rating counts as green at this percentile among the players of the level",
        )
        hitters = first.slider("Green key ratings needed, hitters (of 6)", 1, 6, base.min_green_hitters)
        pitchers = first.slider("Green key ratings needed, pitchers (of 3)", 1, 3, base.min_green_pitchers)
        floor = second.slider(
            "Chance at the next level: mean percentile",
            0.1,
            0.6,
            base.next_floor,
            0.05,
            help="He should not be at the bottom of the level above",
        )
        old = second.slider("Old for the level from age percentile", 0.5, 0.95, base.old_for_level, 0.05)
        prospect = second.slider(
            "Prospect from ceiling percentile",
            0.5,
            0.95,
            base.prospect,
            0.05,
            help="Ceiling among all minor leaguers of the league (hitters and pitchers apart); the better ones are the focus",
        )
        average = third.slider(
            "Average or better: mean percentile",
            0.3,
            0.7,
            base.average,
            0.05,
            help="Needed to move up a player for being old for the level",
        )
        red = third.slider("Weak for the level: mean percentile at most", 0.05, 0.45, base.red, 0.05)
        starters = first.slider("Starting pitchers wanted at every level", 3, 10, base.starters)
        slack = second.slider("Pitchers more or fewer than the level's target", 0, 3, base.pitcher_slack)
        gap = third.slider(
            "Position flagged when another one adds (runs)", 0.0, 10.0, base.position_gap, 0.5,
            help="A hitter whose listed position differs from where the level's best lineup plays him",
        )  # fmt: skip
    return replace(
        base, green=green, min_green_hitters=hitters, min_green_pitchers=pitchers, next_floor=floor, old_for_level=old,
        average=average, red=red, prospect=prospect, top_prospect=max(prospect, base.top_prospect),
        starters=starters, pitcher_slack=slack, position_gap=gap,
    )  # fmt: skip


ACTION_COLORS = {
    roster.PROMOTE: "#2e9e57", roster.DEMOTE: "#d6453d", roster.DFA: "#9b1c1c", roster.RELEASE: "#9b1c1c",
    roster.DFA_OR_RELEASE: "#9b1c1c",
}  # fmt: skip
SECTIONS = (  # title, the actions of the section
    ("Promote", (roster.PROMOTE,)),
    ("Demote", (roster.DEMOTE,)),
    ("DFA or release", (roster.DFA, roster.RELEASE, roster.DFA_OR_RELEASE)),
)


def styled(frame: pd.DataFrame):
    """Whole numbers for the percentiles and the ceiling columns with one decimal; the tier and the action in their colors."""
    style = frame.style.format(
        {
            "Ceil": "{:.1f}",
            "Floor": "{:.1f}",
            "Top-50 %": "{:.0f}",
            "Mean %": "{:.0f}",
            "Gain": "{:.1f}",
            "Age": "{:.0f}",
            "Value": "{:.1f}",
            "Dev": "{:.1f}",
            "Learn": "{:.1f}",
            "Hit": "{:.1f}",
            "Now": "{:.1f}",
            "Grow": "{:+.1f}",
        },
        na_rep="",
    )
    if {"Slot", "Primary"} <= set(frame.columns):  # a slot that is not his own position
        off = frame["Slot"].isin(lineup.LINEUP) & (frame["Slot"] != frame["Primary"])
        style = style.apply(lambda col: ["color: #e08a1e; font-weight: 700" if o else "" for o in off], subset=["Slot"])
    for column, colors in (("Tier", TIER_COLORS), ("Action", ACTION_COLORS)):
        if column in frame:
            style = style.map(lambda v, c=colors: f"color: {c[v]}; font-weight: 700" if v in c else "", subset=[column])
    return style


def move_table(group: pd.DataFrame) -> pd.DataFrame:
    """The columns of a list of moves, the best prospects first (the moves of the prospects with development left matter most)."""
    out = group.drop(
        columns=["To", "Reason"]
    ).copy()  # the wishes of the ratings; the plan's destination and reason follow
    out["Pos"] = out["Listed"] if "Listed" in out else out["Pos"]
    out["Green"] = out["green"].astype(int).astype(str) + "/" + out["pitcher"].map({True: "3", False: "6"})
    out["Mean %"] = out["mean"] * 100
    if "OPT" in out:  # option years used of 3, and the demotions this season (the players of the majors)
        out["Opt"] = [
            f"{int(o)}/3, {int(y) if pd.notna(y) else 0}x" if pd.notna(o) and level == minors.TOP else ""
            for o, y, level in zip(out["OPT"], out["OY"], out["Level"])
        ]
    out = out.rename(columns={"Window": "Win", "Top50": "Top-50 %", "Dest": "To", "Why": "Reason"})
    out["To"] = out["To"].where(out["To"] != "", "out")
    out = out.sort_values(["focus", "ceiling_pct"], ascending=[False, False])
    columns = [
        "ID",
        "Player",
        "Pos",
        "Age",
        "Pro",
        "Tier",
        "Win",
        "Ceil",
        "Floor",
        "Top-50 %",
        "Green",
        "Mean %",
        *(["Opt"] if "Opt" in out and (out["Opt"] != "").any() else []),
        "To",
        "Reason",
    ]
    return out[columns].reset_index(drop=True)


SLOT_ORDER = {slot: k for k, slot in enumerate([*lineup.LINEUP, "BENCH", "SP", "RP"])}


def lineup_table(group: pd.DataFrame) -> pd.DataFrame:
    """Who plays at a level: the lineup slots, then the bench, the rotation and the bullpen, with the parts of the playing value
    (development, learning, the cost of an unfamiliar position, current value) and the words for the place."""
    out = group.assign(order=group["Slot"].map(SLOT_ORDER)).sort_values(["order", "total"], ascending=[True, False])
    out = out.rename(
        columns={"total": "Value", "development": "Dev", "learning": "Learn", "hit": "Hit", "current": "Now"}
    )
    columns = ["ID", "Slot", "Player", "Age", "Primary", "Value", "Dev", "Learn", "Hit", "Now"]
    if "Grow" in out:
        columns.append("Grow")
    return out[[*columns, "Reason"]].reset_index(drop=True)


def position_table(group: pd.DataFrame) -> pd.DataFrame:
    return group.rename(columns={"Better": "Better", "Why": "Reason"})[
        ["ID", "Player", "Age", "Listed", "Better", "Gain", "Reason"]
    ].reset_index(drop=True)


def show(frame: pd.DataFrame, key: str):
    shown = [c for c in frame.columns if c != "ID"]
    config = {
        c: (st.column_config.TextColumn if c in TEXT_COLUMNS else st.column_config.NumberColumn)(
            width=WIDTHS.get(c, 56)
        )
        for c in shown
    }
    event = st.dataframe(
        styled(frame), hide_index=True, column_order=shown, height=min(len(frame), 14) * ROW_HEIGHT + 38,
        row_height=ROW_HEIGHT - 1, on_select="rerun", selection_mode="single-cell", key=key, column_config=config,
    )  # fmt: skip
    picked = event.selection.cells
    return int(frame["ID"].iloc[int(picked[0][0])]) if picked else None


@st.cache_data(show_spinner="Working out the moves...", max_entries=4, ttl=3600)
def planned(
    save: str, season: int, org: str, rules: minors.Rules, version: str, mlb_limit: int, settings: playing.Playing
) -> tuple[roster.Plan, pd.DataFrame]:
    """The plan of the organization and, for the levels it ends up with, who plays where (viewer/lineup.py). The levels are chosen
    together with the slots when `settings.joint_weight` is above 0."""
    frame, ratings, growth = load(save, season)
    levels = get_save(save).levels
    mlb = roster.mlb_table(frame, ratings, org, rules, version)
    peers = roster.mlb_peers(frame, ratings, rules)
    table = minors.manage(frame, ratings, org, levels, rules, version)
    joint = lineup.joint(frame, ratings, rules, settings, growth=growth) if settings.joint_weight > 0 else None
    result = roster.plan(table, frame, levels, rules, mlb, mlb_limit, peers, playing=joint)
    placed = lineup.lineups(result.players, frame, ratings, levels, rules, settings, growth)
    if growth is not None and not placed.empty:
        placed = placed.merge(growth[["ID", "d_cur"]].rename(columns={"d_cur": "Grow"}), on="ID", how="left")
    return result, placed


def playing_from_page(base: playing.Playing) -> playing.Playing:
    """The settings of config/minors.toml [playing], with the sliders of the page on top (for this session only)."""
    with st.expander("Who plays (defaults from config/minors.toml; changes here last for this session)"):
        first, second, third = st.columns(3)
        joint = first.slider(
            "Weight of who plays in the choice of the levels", 0.0, 2.0, base.joint_weight, 0.1,
            help="0: the levels are chosen as before and the lineups only follow; higher: a prospect goes where he plays",
        )  # fmt: skip
        development = first.slider(
            "Development", 0.0, 20.0, base.development, 1.0,
            help="The value of the best prospect (ceiling 1, five years of window, young for the level) in a full-time slot",
        )  # fmt: skip
        learning = second.slider(
            "Position learning (per rating point)", 0.0, 2.0, base.learning, 0.1,
            help="Worth of a point gained at a position by playing it: more at his own position, little at another before "
            "his own is trained",
        )  # fmt: skip
        ready = second.slider(
            "Own position trained before another counts", 0.5, 1.0, base.primary_ready, 0.05,
            help="How close to its potential (0 = 20, 1 = the potential) his position must be; C, SS and CF use the key level",
        )  # fmt: skip
        current = third.slider(
            "Current value",
            0.0,
            20.0,
            base.current,
            1.0,
            help="The floor: the best player of the level now, in a full-time slot",
        )
        hit = third.slider(
            "Performance cost of an unfamiliar position", 0.0, 1.0, base.unfamiliar_hit, 0.05,
            help="Share of the development value lost while he learns a position, less as he learns it",
        )  # fmt: skip
    return replace(
        base, joint_weight=joint, development=development, learning=learning, primary_ready=ready,
        key_ready=max(ready, base.key_ready), current=current, unfamiliar_hit=hit,
    )  # fmt: skip


def choose_mlb_limit(frame: pd.DataFrame, rules: minors.Rules, key: str, phase: str | None = None) -> int:
    """The phase of the season, which sets the active roster of the majors (26, 40 or 60). Starts at the `phase` of the league
    in config/update.toml, else at what the league's teams have on their active rosters."""
    phases = {
        f"Season and playoffs ({rules.mlb_season})": rules.mlb_season,
        f"Offseason ({rules.mlb_offseason})": rules.mlb_offseason,
        f"Spring training ({rules.mlb_spring})": rules.mlb_spring,
    }
    configured = {"season": rules.mlb_season, "offseason": rules.mlb_offseason, "spring": rules.mlb_spring}.get(
        phase or ""
    )
    detected = roster.detect_limit(frame, rules)
    start = configured or detected
    index = list(phases.values()).index(start) if start in phases.values() else 0
    label = st.radio("MLB active roster", list(phases), index=index, horizontal=True, key=key)
    if configured:
        st.caption(f"Starts at the phase of the league in config/update.toml ({phase}).")
    else:
        st.caption(
            f"Guessed from the teams' active rosters: {detected}."
            if detected
            else "No active roster flag (ACT) in the export."
        )
    return phases[label]


NEEDS = {  # what a level needs to field a team (viewer/roster.py), by summary column
    "Catchers": lambda row: roster.NEEDS["C"], "Infielders": lambda row: roster.NEEDS["IF"],
    "Outfielders": lambda row: roster.NEEDS["OF"], "Can start": lambda row: row["Wanted SP"],
}  # fmt: skip


def style_summary(summary: pd.DataFrame, slack: int):
    def over(row):
        limit = row["Max"]
        pitchers = row["Target"]
        return [
            "color: #d6453d; font-weight: 700" if c == "After" and limit and row["After"] > limit else
            "color: #e08a1e; font-weight: 700" if c == "Pitchers after" and pitchers and abs(row[c] - pitchers) > slack else
            "color: #e08a1e; font-weight: 700" if c == "SP after" and row[c] != row["Wanted SP"] else
            "color: #d6453d; font-weight: 700" if c in NEEDS and row[c] < NEEDS[c](row) else ""
            for c in row.index
        ]  # fmt: skip

    return summary.style.apply(over, axis=1)


def main() -> None:
    config = load_update_config()
    st.html(org_view.STYLE)
    league_column, first, *links = st.columns([0.9, 1.2, *org_view.LINK_WIDTHS], vertical_alignment="center")
    save = league.choose_save(config, league_column, key="league-minors-management")
    seasons = served_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = seasons[-1]
    try:
        frame, ratings, _ = load(save, season)
    except FileNotFoundError:
        st.error(f"{save} {season} has no outlook files yet: run `python scripts/update_all.py {save}`.")
        return
    orgs = overview.organizations(frame)
    mine = config[save].get("org")
    org = first.selectbox(
        "Organization", orgs, index=orgs.index(mine) if mine in orgs else 0, label_visibility="collapsed"
    )
    for column, (name, (target, label, icon)) in zip(links, org_view.PAGES.items()):
        try:
            column.page_link(target, label=label, icon=icon, disabled=name == "management")
        except StreamlitPageNotFoundError:  # run on its own, without the navigation of viewer/app.py
            break
    levels = get_save(save).levels
    if not minors.ladder(levels):
        st.error(f"{save} has no level tables in config/saves.toml: they give the promotion ladder.")
        return
    rules = rules_from_page(minors.load_rules())
    version = scales.version_of(get_save(save).ratings)
    mlb_limit = choose_mlb_limit(frame, rules, f"mlb-phase-{save}", config[save].get("phase"))
    settings = playing_from_page(playing.load_playing())
    result, placed = planned(save, season, org, rules, version, mlb_limit, settings)
    table = result.players
    if table.empty:
        st.info(f"{org} has no minor leaguers on the ladder.")
        return
    prospect_share, top_share = 1 - rules.prospect, 1 - rules.top_prospect
    has_limits = any(limits.max_players for limits in levels.values())
    sentences = [
        f"**{org}** {save} {season}: the focus is the development of the better players while they can still improve.",
        f"A prospect has a ceiling in the top {prospect_share:.0%} of the league's minor leaguers "
        f"(top {top_share:.0%}: top prospect). Win = seasons of development left.",
        "Key ratings are compared with the players of each level in all organizations "
        f"(percentile; green from {rules.green:.0%}; Green = how many of the key ratings).",
        "Promote = dominates the level and has a chance at the next one (a prospect with development left needs one green "
        "rating less) or is old for the level, or is over the age or pro-years limit. Demote = weak for the level.",
        "Every move respects the limits of the levels (age, pro years, the most players, the pitchers of a level within "
        f"{rules.pitcher_slack} of its target): a level that is too full sends its fillers down, and a player no level can take "
        "is DFA (on the 40-man roster) or released, the one with the least reason to be kept first.",
        "Who plays: the levels are chosen together with who plays where (a slot at a level is worth its development, the rating "
        "he builds at the position, the cost of an unfamiliar position and his current value; the position he is trained at "
        "comes first, then he moves around). Listed position = the best lineup of the level (one per position, the most "
        "total value) plays him elsewhere; "
        f"a level wants {rules.starters} starting pitchers, relievers who belong in the rotation are flagged.",
        "Click a player for his ratings.",
    ]
    st.caption(" ".join(sentences))
    if not has_limits:
        st.info(
            "The levels have no roster limits yet: set `max_players` and `pitchers` per level in config/saves.toml."
        )
    ladder = minors.ladder(levels)
    present = [lv for lv in [minors.TOP, *ladder] if lv in set(table["Level"])]
    st.dataframe(style_summary(result.summary, rules.pitcher_slack), height=min(len(result.summary), 10) * 35 + 38)
    if not placed.empty:
        st.caption(
            "Who plays: players placed at the level they end up at; off primary = at a slot other than the position he is "
            "trained at; differs from chart = the depth chart (most current value per position) would place him elsewhere; "
            "playing value = the sum of the values of the lineup and the rotation."
        )
        st.dataframe(lineup.comparison(placed), hide_index=True, height=min(len(present), 10) * 35 + 38)
    generation = st.session_state.setdefault("minors_generation", 0)
    picked = None
    for level in present:
        here = table[table["Level"] == level]
        st.subheader(f"{level}: {len(here)} players, {int((here['Action'] != '').sum())} to move")
        shown_any = False
        for title, actions in SECTIONS:
            group = here[here["Action"].isin(actions)]
            if group.empty:
                continue
            shown_any = True
            st.markdown(f"**{title}** ({len(group)})")
            clicked = show(move_table(group), f"minors-{save}-{org}-{level}-{title}-{generation}")
            if clicked and not picked:
                row = group.loc[group["ID"] == clicked].iloc[0]
                picked = (clicked, row["Pos"], row["Player"], level)
        wrong = result.positions[result.positions["Level"] == level]
        if not wrong.empty:
            shown_any = True
            st.markdown(f"**Listed at a position that is not their best** ({len(wrong)})")
            clicked = show(position_table(wrong), f"minors-{save}-{org}-{level}-position-{generation}")
            if clicked and not picked:
                row = here.loc[here["ID"] == clicked].iloc[0]
                picked = (clicked, row["Pos"], row["Player"], level)
        if not shown_any:
            st.write("No moves.")
        if level != minors.TOP and not placed.empty:
            playing_here = placed[placed["Level"] == level]
            if not playing_here.empty:
                st.markdown(f"**Who plays** ({len(playing_here)}): the lineup, the bench, the rotation and the bullpen")
                clicked = show(lineup_table(playing_here), f"minors-{save}-{org}-{level}-plays-{generation}")
                if clicked and not picked:
                    row = here.loc[here["ID"] == clicked]
                    if not row.empty:
                        picked = (clicked, row.iloc[0]["Pos"], row.iloc[0]["Player"], level)
    if picked:
        st.session_state["minors_generation"] = generation + 1
        player_id, slot, name, level = picked
        org_view.player_dialog(save, season, player_id, slot, name, f"{org}, {level}, {slot}")


main()
