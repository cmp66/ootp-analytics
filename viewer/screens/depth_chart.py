"""The depth chart screen, modelled on the organizational depth chart of OOTP.

A field with a box per position: the best five at each fielding position and the best ten starting and relief pitchers of one
organization, each player once, placed the way a manager would build the team. Choices: the organization, the part of it to show
(the entire organization or the minor leagues) and the ranking (current value, or the ceiling / floor of the best season in 10).
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from scripts.update_all import load_update_config  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from viewer import depth, overview, queries  # noqa: E402
from viewer.screens import org_view  # noqa: E402

BOX_TITLES = {
    "LF": "Left Field", "CF": "Center Field", "RF": "Right Field", "3B": "Third Base", "SS": "Shortstop",
    "2B": "Second Base", "1B": "First Base", "C": "Catcher", "SP": "Starting Rotation", "RP": "Bullpen",
}  # fmt: skip
# where the box of a position sits on the field: (centre x %, top %), after the OOTP chart
BOX_PLACES = {
    "LF": (22, 17), "CF": (52, 14), "RF": (82, 17), "SS": (40, 42), "2B": (65, 42), "3B": (14, 46), "1B": (89, 46),
    "SP": (20, 65), "C": (52, 70), "RP": (81, 64),
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
.box .none {{color: #8a93a8; font-style: italic;}}
</style>
"""


def over(row) -> str:
    """Red name with the reason as a tooltip for a player who exceeds the limits of his level."""
    return (
        f' class="over" title="Over the limits of {html.escape(str(row.Lev))}: {html.escape(row.Flag)}"'
        if row.Flag
        else ""
    )


def box_html(slot: str, players) -> str:
    left, top = BOX_PLACES[slot]
    rows = (
        "".join(
            f'<div class="row"><span{over(row)}>{html.escape(row.Player)}</span>'
            f'<span class="lev">{html.escape(str(row.Lev))}</span><span class="value">{row.Value:.1f}</span></div>'
            for row in players.itertuples()
        )
        or '<div class="none">nobody</div>'
    )
    kind = " pitchers" if slot in ("SP", "RP") else ""
    return f'<div class="box{kind}" style="left:{left}%; top:{top}%"><h4>{BOX_TITLES[slot]}</h4>{rows}</div>'


def chart_html(chart: dict) -> str:
    boxes = "".join(box_html(slot, players) for slot, players in chart.items())
    return f'<div class="chart"><div class="field"></div><div class="dirt"></div>{boxes}</div>'


@st.cache_data(show_spinner="Loading the player files...")
def load(save: str, season: int):
    return queries.load_outlook(save, season)


def main() -> None:
    config = load_update_config()
    save = next(iter(config))
    seasons = available_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = seasons[-1]
    try:
        frame = load(save, season)
    except FileNotFoundError:
        st.error(f"{save} {season} has no outlook files yet: run `python scripts/update_all.py {save}`.")
        return
    st.html(STYLE)
    orgs = overview.organizations(frame)
    mine = config[save].get("org")
    first, show, ranked, *links = st.columns([1.2, 1.4, 1.6, 1, 1.3, 1, 0.8, 1, 1.1], vertical_alignment="center")
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
    try:
        chart = depth.depth_chart(frame, org, choice, ranking, get_save(save).levels)
    except ValueError as error:  # outlook files made before the career view have no 10-season floor / ceiling
        st.error(str(error))
        return
    st.caption(
        f"**{org}** {save} {season}: built like a team: the best lineup is picked first, every player at the position where he "
        f"adds the most given the others, then the best backups from the players left, and so on; each player once, "
        f"{depth.DEPTH_HITTERS} deep at each position, {depth.DEPTH_PITCHERS} at starting and relief pitcher. "
        "A name in red is over the age or pro-years limit of his level (config/saves.toml; hover for the reason). "
        + {
            "Current": "Ranked by value now (runs above the league's average starter at the position).",
            "Ceiling (10 years)": "Ranked by the ceiling: the 90th percentile of his best season within the next 10.",
            "Floor (10 years)": "Ranked by the floor: the 10th percentile of his best season within the next 10.",
        }[ranking]
    )
    st.html(chart_html(chart))


main()
