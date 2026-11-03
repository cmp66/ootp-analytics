"""The roster acquisition page: where the MLB roster of an organization needs an upgrade and who could provide it.

The needs are the places of the starting lineup, rotation and bullpen where the starter is weak against the starters of the
league; the candidates are the organization's own players who are not starters (internal), players on waivers and free agents,
with what they cost against the budget space of the season. See config/acquisition.toml and viewer/acquire.py.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from scripts.update_all import load_update_config  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from viewer import acquire, league, queries  # noqa: E402
from viewer.screens import org_view  # noqa: E402

ROW_HEIGHT = 35
WIDTHS = {
    "Place": 60, "Player": 150, "Age": 50, "Value": 62, "Park": 56, "Pct": 56, "Target": 62, "Gap": 56, "Need": 50, "Source": 90,
    "From": 60, "Deal": 95, "Cost $m": 70, "Sign": 100, "Yrs": 50, "Gain": 56, "Per $m": 62, "Best option": 190, "Option": 190,
}  # fmt: skip
TEXT_COLUMNS = ("Place", "Player", "Need", "Source", "From", "Deal", "Sign", "Yrs", "Best option", "Option")


@st.cache_data(show_spinner="Loading the player files...")
def load(save: str, season: int):
    return queries.load_outlook(save, season), queries.load_finances(save, season)


@st.cache_data(show_spinner="Working out the rosters...")
def compute(save: str, season: int, org: str, settings: acquire.Settings, park_on: bool = False):
    """The needs and the candidates. With the park adjustment every team's starters are valued in their own park (the fair
    comparison) and the candidates, wherever they come from, in the park of `org`."""
    frame, _ = load(save, season)
    needs = acquire.needs(league.park_frame(save, season, frame, org, park_on, own=True), org, settings)
    pool = acquire.candidate_pool(league.park_frame(save, season, frame, org, park_on), org, needs, settings)
    return needs, pool


def settings_from_page(base: acquire.Settings) -> acquire.Settings:
    with st.expander("Rules (defaults from config/acquisition.toml; changes here last for this session)"):
        first, second, third = st.columns(3)
        need = first.slider(
            "A starter is a need at or below percentile", 0.1, 0.7, base.need_percentile, 0.05,
            help="Where he stands among the league's starters at the place",
        )  # fmt: skip
        target = first.slider("Target: percentile of the league's starters", 0.3, 0.9, base.target_percentile, 0.05)
        gain = second.slider("Least gain to list a candidate (runs)", 0.0, 10.0, base.min_gain, 0.5)
        pen = second.slider("Relievers counted as starters", 3, 10, base.bullpen)
        relief = third.slider("Weight of a run of a reliever", 0.1, 1.0, dict(base.weights).get("RP", 1.0), 0.05)
        hard = third.multiselect(
            "Free agents not listed when their signability is",
            ["Impossible", "Extremely Hard", "Hard", "Normal", "Easy", "Very Easy"],
            list(base.hard_to_sign),
        )
    return replace(
        base, need_percentile=need, target_percentile=target, min_gain=gain, bullpen=pen, weights=(("RP", relief),),
        hard_to_sign=tuple(hard),
    )  # fmt: skip


def money(value) -> str:
    return "" if pd.isna(value) else f"${value / 1e6:.1f}m"


def option_text(row: pd.Series | None) -> str:
    return "" if row is None else f"{row['Player']} ({row['Source']}, {money(row['Cost'])}, +{row['Gain']:.1f})"


def needs_table(needs: pd.DataFrame, pool: pd.DataFrame, budget: float, settings: acquire.Settings) -> pd.DataFrame:
    """The places of the roster, the biggest needs first, with the best option within the budget space for each need."""
    best = []
    for _, row in needs.iterrows():
        found = acquire.options(pool, row, settings)
        found = found[found["Cost"] <= budget] if pd.notna(budget) else found
        best.append(option_text(found.iloc[0]) if row["need"] and len(found) else "")
    out = pd.DataFrame(
        {
            "Place": needs["place"], "Player": needs["Player"], "Age": needs["Age"], "Value": needs["value"],
            **({"Park": needs["park"]} if "park" in needs else {}), "Pct": needs["pct"] * 100, "Target": needs["target"],
            "Gap": needs["gap"],
            "Need": np.where(needs["need"], "NEED", ""), "Best option": best,
        }
    )  # fmt: skip
    return out


def style_needs(frame: pd.DataFrame):
    style = frame.style.format(
        {"Value": "{:.1f}", "Park": "{:+.1f}", "Target": "{:.1f}", "Gap": "{:.1f}", "Pct": "{:.0f}", "Age": "{:.0f}"},
        na_rep="",
    )
    return style.apply(lambda row: ["font-weight: 700; color: #d6453d" if row["Need"] else "" for _ in row], axis=1)


def show(frame: pd.DataFrame, styled, key: str, height_rows: int = 14):
    """A table with the ID kept out of sight; returns the ID of the player clicked, if any."""
    shown = [c for c in frame.columns if c not in ("ID", "slot")]
    config = {
        c: (st.column_config.TextColumn if c in TEXT_COLUMNS else st.column_config.NumberColumn)(
            width=WIDTHS.get(c, 60)
        )
        for c in shown
    }
    event = st.dataframe(
        styled, hide_index=True, column_order=shown, height=min(len(frame), height_rows) * ROW_HEIGHT + 38,
        row_height=ROW_HEIGHT - 1,
        on_select="rerun", selection_mode="single-cell", key=key, column_config=config,
    )  # fmt: skip
    picked = event.selection.cells
    if not picked or "ID" not in frame:
        return None
    row = frame.iloc[int(picked[0][0])]
    return (int(row["ID"]), str(row["slot"]), str(row["Player"])) if pd.notna(row["ID"]) else None


def candidate_view(options: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ID": options["ID"], "slot": options["slot"], "Player": options["Player"], "Age": options["Age"],
            "Source": options["Source"], "From": options["From"], "Deal": options["Deal"],
            "Cost $m": options["Cost"] / 1e6, "Sign": options["Sign"],
            "Yrs": options["Years"].astype(str), "Value": options["value"],
            **({"Park": options["park"]} if "park" in options else {}), "Gain": options["Gain"],
            "Per $m": options["Gain"] / (options["Cost"] / 1e6).clip(lower=0.5),
        }
    )  # fmt: skip
    return out


def style_candidates(frame: pd.DataFrame):
    formats = {
        "Cost $m": "{:.1f}",
        "Value": "{:.1f}",
        "Park": "{:+.1f}",
        "Gain": "{:.1f}",
        "Per $m": "{:.1f}",
        "Age": "{:.0f}",
    }
    return frame.style.format(formats, na_rep="")


def main() -> None:
    config = load_update_config()
    st.html(org_view.STYLE)
    league_column, first, *links = st.columns([0.9, 1.2, *org_view.LINK_WIDTHS], vertical_alignment="center")
    save = league.choose_save(config, league_column, key="league-acquire")
    seasons = available_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = seasons[-1]
    try:
        frame, finances = load(save, season)
    except FileNotFoundError:
        st.error(f"{save} {season} has no outlook files yet: run `python scripts/update_all.py {save}`.")
        return
    orgs = sorted(set(acquire.mlb_players(frame)["org"]))
    mine = config[save].get("org")
    org = first.selectbox(
        "Organization", orgs, index=orgs.index(mine) if mine in orgs else 0, label_visibility="collapsed"
    )
    for column, (name, (target, label, icon)) in zip(links, org_view.PAGES.items()):
        try:
            column.page_link(target, label=label, icon=icon, disabled=name == "acquire")
        except StreamlitPageNotFoundError:  # run on its own, without the navigation of viewer/app.py
            break
    settings = settings_from_page(acquire.load_settings())
    park_on = league.park_toggle(st, org, "park-acquire", league.park_available(save, season))
    needs, pool = compute(save, season, org, settings, park_on)
    budget = acquire.budget_space(finances, org)

    metrics = st.columns(4)
    metrics[0].metric("Budget space", money(budget) if pd.notna(budget) else "unknown")
    team = finances[finances["Team"] == str(org)] if finances is not None else pd.DataFrame()
    metrics[1].metric("Payroll", money(team["Payroll"].iloc[0]) if len(team) else "unknown")
    metrics[2].metric("Places that need an upgrade", int(needs["need"].sum()))
    metrics[3].metric("Strategy", str(team["Mode"].iloc[0]) if len(team) and "Mode" in team else "unknown")
    if finances is None:
        st.info("This league has no finance files: costs are not checked against a budget.")
    st.caption(
        f"**{org}** {save} {season}: the starters are the best lineup, five starters and {settings.bullpen} relievers of the MLB "
        "roster. Pct = where the starter stands among the league's starters at the place; a NEED is at or below "
        f"{settings.need_percentile:.0%}. Candidates: your own players who are not starters (internal), players on waivers "
        "(claimed with their contract) and free agents (what they ask for a year). Gain = runs of value over the starter, "
        "weighted by the place."
    )
    table = needs_table(needs, pool, budget, settings)
    st.subheader("Where the roster needs an upgrade")
    st.dataframe(
        style_needs(table), hide_index=True, height=min(len(table), 22) * (ROW_HEIGHT - 1) + 38, row_height=ROW_HEIGHT - 1,
        column_config={
            c: (st.column_config.TextColumn if c in TEXT_COLUMNS else st.column_config.NumberColumn)(width=WIDTHS.get(c, 60))
            for c in table.columns
        },
    )  # fmt: skip

    st.subheader("Best set of acquisitions within the budget space")
    everywhere = st.checkbox("Also upgrade places that are not a need")
    chosen = acquire.best_package(
        needs if everywhere else needs[needs["need"]], pool, budget if pd.notna(budget) else 1e12, settings
    )
    if chosen.empty:
        st.write("No candidate adds enough within the budget space.")
    else:
        view = candidate_view(chosen)
        view.insert(0, "Place", chosen["Place"])
        st.dataframe(
            style_candidates(view), hide_index=True, column_order=[c for c in view.columns if c not in ("ID", "slot")]
        )
        spent = chosen["Cost"].sum()
        left = budget - spent if pd.notna(budget) else np.nan
        st.caption(
            f"{len(chosen)} acquisitions, {money(spent)}, total gain {chosen['Gain'].sum():.1f} runs"
            + (f"; budget space left {money(left)}." if pd.notna(left) else ".")
        )

    st.subheader("Candidates for a place")
    ordered = list(needs["place"])
    place = st.selectbox(
        "Place", ordered, index=0, format_func=lambda p: f"{p}: {needs.set_index('place').loc[p, 'Player']}"
    )
    row = needs.set_index("place").loc[place].rename("row")
    row["place"] = place
    sources = st.multiselect("Sources", list(acquire.SOURCES), list(acquire.SOURCES))
    within = st.checkbox("Only candidates within the budget space", value=True)
    found = acquire.options(pool, row, settings)
    found = found[found["Source"].isin(sources)]
    if within and pd.notna(budget):
        found = found[found["Cost"] <= budget]
    if found.empty:
        st.write("No candidate adds enough at this place with these choices.")
    else:
        view = candidate_view(found.head(40))
        generation = st.session_state.setdefault("acquire_generation", 0)
        clicked = show(view, style_candidates(view), f"acquire-{save}-{org}-{place}-{sources}-{within}-{generation}")
        if clicked:
            st.session_state["acquire_generation"] = generation + 1
            player_id, slot, name = clicked
            org_view.player_dialog(save, season, player_id, slot, name, f"{org}, candidate for {place}")


main()
