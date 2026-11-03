"""The minor league management page: which players of an organization should move up or down a level.

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

from scripts.update_all import load_update_config  # noqa: E402
from util import paths  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from viewer import league, minors, overview, queries  # noqa: E402
from viewer.screens import org_view  # noqa: E402

ROW_HEIGHT = 35
WIDTHS = {"Player": 150, "Pos": 50, "Age": 50, "Pro": 50, "Move": 90, "To": 60, "Reason": 420, "Blocked": 190}
MOVE_COLORS = {minors.UP: "#2e9e57", minors.DOWN: "#d6453d", minors.MUST: "#e08a1e"}
ORDER = {minors.MUST: 0, minors.UP: 1, minors.DOWN: 2, "": 3}


@st.cache_data(show_spinner="Loading the player files...")
def load(save: str, season: int):
    ratings = pd.read_csv(paths.output_file(save, season, "player-data"), low_memory=False)
    return queries.load_outlook(save, season), ratings


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
        average = third.slider(
            "Average or better: mean percentile",
            0.3,
            0.7,
            base.average,
            0.05,
            help="Needed to move up a player for being old for the level",
        )
        red = third.slider("Weak for the level: mean percentile at most", 0.05, 0.45, base.red, 0.05)
    return replace(
        base, green=green, min_green_hitters=hitters, min_green_pitchers=pitchers, next_floor=floor, old_for_level=old,
        average=average, red=red,
    )  # fmt: skip


def styled(frame: pd.DataFrame, rules: minors.Rules, keys: list[str]):
    """Percentiles as whole numbers: green from the rule, red for the bottom quarter; the move in its color."""
    labels = [minors.LABELS[k] for k in keys]
    style = frame.style.format({c: "{:.0f}" for c in labels}, na_rep="")
    style = style.map(
        lambda v: (
            "color: #2e9e57; font-weight: 700"
            if pd.notna(v) and v >= rules.green * 100
            else "color: #d6453d" if pd.notna(v) and v <= rules.red * 100 else ""
        ),
        subset=labels,
    )
    return style.map(
        lambda v: f"color: {MOVE_COLORS[v]}; font-weight: 700" if v in MOVE_COLORS else "", subset=["Move"]
    )


def level_table(table: pd.DataFrame, keys: list[str], rules: minors.Rules) -> pd.DataFrame:
    """The columns of the page for the players of one level and kind, the ratings as percentiles (0 to 100)."""
    out = table.copy()
    for key in keys:
        out[minors.LABELS[key]] = out[f"pct_{key}"] * 100
    out["Move"] = out["Move"].replace("", None)
    out["_order"] = out["Move"].fillna("").map(ORDER)
    out = out.sort_values(["_order", "mean"], ascending=[True, False]).drop(columns="_order")
    out["To"] = out["To"].fillna("")
    out["Move"] = out["Move"].fillna("")
    columns = [
        "ID",
        "Player",
        "Pos",
        "Age",
        "Pro",
        *[minors.LABELS[k] for k in keys],
        "Move",
        "To",
        "Reason",
        "Blocked",
    ]
    return out[columns].reset_index(drop=True)


def show(frame: pd.DataFrame, keys: list[str], rules: minors.Rules, key: str):
    shown = [c for c in frame.columns if c != "ID"]
    config = {
        c: (
            st.column_config.TextColumn
            if c in ("Player", "Pos", "Move", "To", "Reason", "Blocked")
            else st.column_config.NumberColumn
        )(width=WIDTHS.get(c, 56))
        for c in shown
    }
    event = st.dataframe(
        styled(frame, rules, keys), hide_index=True, column_order=shown, height=min(len(frame), 14) * ROW_HEIGHT + 38,
        row_height=ROW_HEIGHT - 1, on_select="rerun", selection_mode="single-cell", key=key, column_config=config,
    )  # fmt: skip
    picked = event.selection.cells
    return int(frame["ID"].iloc[int(picked[0][0])]) if picked else None


def main() -> None:
    config = load_update_config()
    st.html(org_view.STYLE)
    league_column, first, *links = st.columns([0.9, 1.2, 1, 1.3, 1, 0.8, 1, 1.1, 1.3], vertical_alignment="center")
    save = league.choose_save(config, league_column, key="league-minors-management")
    seasons = available_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = seasons[-1]
    try:
        frame, ratings = load(save, season)
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
    table = minors.manage(frame, ratings, org, levels, rules)
    if table.empty:
        st.info(f"{org} has no minor leaguers on the ladder.")
        return
    st.caption(
        f"**{org}** {save} {season}: key ratings are compared with the players of each level in all organizations (percentile; "
        f"green from {rules.green:.0%}). Up = dominates the level and has a chance at the next one, or is old for the level with "
        "average or better ratings; Down = weak for the level; Must move = over the age or pro-years limit of the level. A move "
        "is only listed to a level whose limits he meets (Blocked tells when it is not). Click a player for his ratings."
    )
    everyone = st.checkbox("Also show players with no move")
    ladder = minors.ladder(levels)
    summary = (
        table.assign(Move=table["Move"].replace("", "-"))
        .pivot_table(index="Level", columns="Move", values="ID", aggfunc="count", fill_value=0)
        .reindex([lv for lv in ladder if lv in set(table["Level"])])
    )
    summary.insert(0, "Players", table.groupby("Level")["ID"].count())
    st.dataframe(summary, height=min(len(summary), 10) * 35 + 38)
    generation = st.session_state.setdefault("minors_generation", 0)
    picked = None
    for level in ladder:
        part = table[table["Level"] == level]
        if part.empty:
            continue
        moves = part[part["Move"] != ""]
        st.subheader(f"{level}: {len(part)} players, {len(moves)} to move")
        for pitcher, kind in ((False, "Hitters"), (True, "Pitchers")):
            group = part[part["pitcher"] == pitcher]
            group = group if everyone else group[group["Move"] != ""]
            if group.empty:
                continue
            keys = [k for k in rules.keys(pitcher) if f"pct_{k}" in group.columns]
            shown = level_table(group, keys, rules)
            st.markdown(f"**{kind}**")
            clicked = show(shown, keys, rules, f"minors-{save}-{org}-{level}-{kind}-{everyone}-{generation}")
            picked = picked or (
                clicked
                and (
                    clicked,
                    group.loc[group["ID"] == clicked, "Pos"].iloc[0],
                    group.loc[group["ID"] == clicked, "Player"].iloc[0],
                    level,
                )
            )
    if picked:
        st.session_state["minors_generation"] = generation + 1
        player_id, slot, name, level = picked
        org_view.player_dialog(save, season, player_id, slot, name, f"{org}, {level}, {slot}")


main()
