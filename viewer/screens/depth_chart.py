"""The depth chart screen, modelled on the organizational depth chart of OOTP.

A field with a box per position: the best five at each fielding position and the best ten starting and relief pitchers of one
organization, each player once, placed the way a manager would build the team. Choices: the organization, the part of it to show
(the entire organization or the minor leagues) and the ranking (current value, or the ceiling / floor of the best season in 10).
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from model import scales  # noqa: E402
from util.update_config import load_update_config  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import served_seasons  # noqa: E402
from viewer import data_cache, depth, league, minors, org_plan, overview, playing, roster  # noqa: E402
from viewer.screens import org_view  # noqa: E402

BOX_TITLES = {
    "LF": "Left Field", "CF": "Center Field", "RF": "Right Field", "3B": "Third Base", "SS": "Shortstop",
    "2B": "Second Base", "1B": "First Base", "C": "Catcher", "DH": "Designated Hitter",
    "SP": "Starting Rotation", "RP": "Bullpen",
}  # fmt: skip
# where the box of a position sits on the field: (centre x %, top %), after the OOTP chart
BOX_PLACES = {
    "LF": (22, 17), "CF": (52, 14), "RF": (82, 17), "SS": (40, 42), "2B": (65, 42), "3B": (14, 46), "1B": (89, 46),
    "SP": (20, 65), "C": (52, 70), "RP": (81, 64), "DH": (38, 87),
}  # fmt: skip
CHART_HEIGHT = 1150  # pixels

STYLE = org_view.STYLE + f"""
<style>
.chart {{position: relative; height: {CHART_HEIGHT}px; background: #1c1c1c; border-radius: 6px; overflow: hidden;}}
.field {{position: absolute; left: 3%; right: 3%; top: 6%; bottom: 2%;
  background: repeating-linear-gradient(45deg, #2f6a35 0 28px, #33723a 28px 56px);
  clip-path: polygon(0 36%, 50% 0, 100% 36%, 100% 58%, 50% 100%, 0 58%);}}
.dirt {{position: absolute; left: 33%; right: 33%; bottom: 3%; height: 30%; border-radius: 50% 50% 0 0;
  background: radial-gradient(circle at 50% 100%, #b4694a 0 45%, #a35d41 70%, transparent 71%);}}
.box {{position: absolute; transform: translateX(-50%); background: rgba(24, 28, 40, 0.82); color: #fff; padding: 2px 8px 4px;
  font-size: 14px; line-height: 1.55; min-width: 200px;}}
.box.pitchers {{min-width: 250px;}}
.box h4 {{margin: 0; font-size: 15px; font-weight: 600; color: #ffe600;}}
.box .row {{display: flex; justify-content: space-between; gap: 12px; white-space: nowrap;}}
.box .lev {{color: #cfd6e6;}}
.box .value {{color: #9fb0cf; font-size: 12px; min-width: 38px; text-align: right;}}
.box .over {{color: #ff7b72;}}
.box .off {{color: #ffb347;}}
.box .none {{color: #8a93a8; font-style: italic;}}
</style>
"""


def style_of(row) -> str:
    """The attributes of a player's name: red with the reason as a tooltip when he exceeds the limits of his level; with a plan,
    orange when he plays a position other than the one he is trained at, and the reason for his place as the tooltip."""
    tips = []
    if row.Flag:
        tips.append(f"Over the limits of {row.Lev}: {row.Flag}")
    if getattr(row, "Note", ""):
        tips.append(row.Note)
    classes = "over" if row.Flag else "off" if getattr(row, "Off", False) else ""
    title = f' title="{html.escape(" | ".join(tips))}"' if tips else ""
    return (f' class="{classes}"' if classes else "") + title


def player_rows(players) -> str:
    return (
        "".join(
            f'<div class="row"><span{style_of(row)}>{html.escape(row.Player)}</span>'
            f'<span class="lev">{html.escape(str(row.Lev))}</span>'
            f'<span class="value">{"" if pd.isna(row.Value) else f"{row.Value:.1f}"}</span></div>'
            for row in players.itertuples()
        )
        or '<div class="none">nobody</div>'
    )


def box_html(slot: str, players) -> str:
    left, top = BOX_PLACES[slot]
    kind = " pitchers" if slot in ("SP", "RP") else ""
    heading = f"<h4>{BOX_TITLES[slot]}</h4>"
    return f'<div class="box{kind}" style="left:{left}%; top:{top}%">{heading}{player_rows(players)}</div>'


def chart_html(chart: dict) -> str:
    boxes = "".join(box_html(slot, players) for slot, players in chart.items())
    return f'<div class="chart"><div class="field"></div><div class="dirt"></div>{boxes}</div>'


def load(save: str, season: int):
    return data_cache.outlook(save, season)


AS_PLANNED, AS_IS = "As planned", "As is"


def planned_from_session(save: str, season: int, org: str, frame: pd.DataFrame):
    """(who plays where, the playing value of every slot) of the plan of the Level moves page: with the rules, the phase of the
    majors and the weights set on that page in this session (the files' defaults otherwise). None without minor leaguers.
    """
    rules, settings, limit = minors.load_rules(), playing.load_playing(), None
    kept = st.session_state.get("planner")
    if kept and kept[:2] == (save, org):
        _, _, rules, limit, settings = kept
    if limit is None:
        phase = load_update_config()[save].get("phase")
        limit = {
            "season": rules.mlb_season, "offseason": rules.mlb_offseason, "spring": rules.mlb_spring
        }.get(phase or "") or roster.detect_limit(frame, rules) or rules.mlb_season  # fmt: skip
    version = scales.version_of(get_save(save).ratings)
    result, placed, offered = org_plan.planned(save, season, org, rules, version, limit, settings)
    return None if placed.empty else (placed, offered)


def main() -> None:
    config = load_update_config()
    st.html(STYLE)
    league_column, first, show, ranked, *links = st.columns(
        [0.9, 1.2, 1.4, 1.6, *org_view.LINK_WIDTHS], vertical_alignment="center"
    )
    save = league.choose_save(config, league_column, key="league-depth")
    seasons = served_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = seasons[-1]
    try:
        frame = load(save, season)
    except FileNotFoundError:
        st.error(f"{save} {season} has no outlook files yet: run `python scripts/update_all.py {save}`.")
        return
    orgs = overview.organizations(frame)
    mine = config[save].get("org")
    org = first.selectbox(
        "Organization", orgs, index=orgs.index(mine) if mine in orgs else 0, label_visibility="collapsed"
    )
    choice = show.selectbox(
        "Show",
        depth.show_options(frame, org),
        label_visibility="collapsed",
        format_func=lambda s: f"Show: {depth.show_label(s)}",
    )
    ranking = ranked.selectbox(
        "Ranked by", depth.RANKED_BY, label_visibility="collapsed", format_func=lambda s: f"Ranked by: {s}"
    )
    for column, (key, (target, label, icon)) in zip(links, org_view.PAGES.items()):
        try:
            column.page_link(target, label=label, icon=icon, disabled=key == "depth")
        except StreamlitPageNotFoundError:  # run on its own, without the navigation of viewer/app.py
            break
    park_column, source_column, side_column, text_column = st.columns([1.9, 1.7, 1.6, 4.8], vertical_alignment="center")
    park_on = league.park_toggle(park_column, org, "park-depth", league.park_available(save, season))
    levels = get_save(save).levels
    sources = [AS_PLANNED, AS_IS] if minors.ladder(levels) else [AS_IS]
    source = source_column.selectbox(
        "Players",
        sources,
        label_visibility="collapsed",
        format_func=lambda s: f"Players: {s}",
        help="As planned: the lineups of the plan on the Level moves page (the levels and who plays where, by playing value). "
        "As is: where the files have the players now, ranked by the value chosen.",
    )
    planned_view = source == AS_PLANNED
    plan = None
    if planned_view:
        plan = planned_from_session(save, season, org, frame)
        if plan is None:
            st.info(f"{org} has no minor leaguers on the ladder: showing the players as they are.")
            planned_view = False
    frame = league.park_frame(save, season, frame, org, park_on)  # his organization's players valued in its park
    splits = (
        ranking == "Current" and "value_vs_right" in frame.columns and not planned_view
    )  # the projections have no split by the pitcher's hand; the plan has none either
    view = side_column.selectbox(
        "Lineup",
        depth.SIDE_CHOICES if splits else depth.SIDE_CHOICES[:1],
        label_visibility="collapsed",
        format_func=lambda s: f"Lineup: {s}",
        help="The hitters' values against right- or left-handed pitchers (current values only).",
        disabled=not splits,
    )
    sides = {label: side for side, label in depth.SIDE_LABELS.items()}
    try:
        if planned_view:
            chart = depth.planned_chart(frame, org, choice, plan[0], plan[1], levels)
        else:
            chart = depth.depth_chart(frame, org, choice, ranking, levels, sides.get(view))
    except ValueError as error:  # outlook files made before the career view have no 10-season floor / ceiling
        st.error(str(error))
        return
    if planned_view:
        text_column.caption(
            f"**{org}** {save} {season}: the plan of the Level moves page. The levels are the ones the plan puts the players at "
            "(the level shown); each starter is the plan's, then the backups from the players of the level by their playing "
            "value; the rotation and the bullpen are the plan's. The whole organization shows the starters of each level from "
            "the top down. A name in orange plays a position other than the one he is trained at; hover for the reason. Red: "
            "over the limits of the level. The value is the value now at the slot (runs above the league's average starter). The "
            "ranking and lineup choices apply to As is."
        )
    else:
        text_column.caption(
            f"**{org}** {save} {season}: built like a team: the best lineup is picked first, every player at the position where "
            f"he adds the most given the others, then the best backups from the players left, and so on; each player once, "
            f"{depth.DEPTH_HITTERS} deep at each position, {depth.DEPTH_PITCHERS} at starting and relief pitcher. "
            "A name in red is over the age or pro-years limit of his level (config/saves.toml; hover for the reason). "
            + {
                "Current": "Ranked by value now (runs above the league's average starter at the position).",
                "Ceiling (10 years)": "Ranked by the ceiling: the 90th percentile of his best season within the next 10.",
                "Floor (10 years)": "Ranked by the floor: the 10th percentile of his best season within the next 10.",
            }[ranking]
            + (
                " The hitters' values are against right- or left-handed pitchers (the batting part of the value swapped for "
                "his split); fielding and running do not change, the pitchers are one list."
                if view != "Overall"
                else ""
            )
        )
    # the chart is drawn as html, which cannot be clicked: a list of its players opens the ratings page of one
    players = {
        f"{slot}: {row.Player} ({row.Lev})": (int(row.ID), slot, row.Player)
        for slot, rows in chart.items()
        for row in rows.itertuples()
    }
    generation = st.session_state.setdefault("depth_generation", 0)  # a new key clears the choice
    chosen = st.columns([3, 7])[0].selectbox(
        "Open a player's ratings",
        [None, *players],
        format_func=lambda c: "Choose a player..." if c is None else c,
        key=f"depth-open-{save}-{org}-{generation}",
    )
    st.html(chart_html(chart))
    if chosen:
        st.session_state["depth_generation"] = generation + 1
        player_id, slot, name = players[chosen]
        org_view.player_dialog(save, season, player_id, slot, name, f"{org}, {slot}")


main()
