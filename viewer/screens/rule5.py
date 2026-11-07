"""The Rule 5 page: which of your draft eligible players to protect on the 40-man roster, who could come off it to make room, and
the players of other organizations you could draft.

The export flags the draft eligible players (R5) and, where the league has it, who is on the 40-man (ON40). See config/rule5.toml
and viewer/rule5.py.
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from model import scales  # noqa: E402
from util.update_config import load_update_config  # noqa: E402
from util.config import get_save  # noqa: E402
from util.seasons import served_seasons  # noqa: E402
from viewer import acquire, data_cache, injuries, league, minors, rule5  # noqa: E402
from viewer.screens import org_view  # noqa: E402

ROW_HEIGHT = 35
WIDTHS = {
    "Inj": 96,
    "Player": 150, "Pos": 50, "Age": 50, "Level": 60, "Tier": 105, "Win": 50, "Value": 62, "MLB %": 62, "Gain": 56, "Upside": 62,
    "Ceil": 56, "Floor": 56, "Top-50 %": 70, "Action": 90, "Why": 380, "Keep %": 64, "Owed $m": 70, "Years": 90, "From": 60,
    "Place": 60, "Go": 60,
}  # fmt: skip
TEXT_COLUMNS = ("Player", "Inj", "Pos", "Level", "Tier", "Action", "Why", "Years", "From", "Place", "Go")
ACTION_COLORS = {rule5.PROTECT: "#2e9e57", rule5.LOW_RISK: "#b8860b", rule5.EXPOSE: "#8a8f99"}


def load(save: str, season: int):
    return data_cache.outlook(save, season)


def money(value) -> str:
    return "" if pd.isna(value) else f"${value / 1e6:.1f}m"


def show(frame: pd.DataFrame, styled, key: str, rows: int = 16):
    """A table with the ID kept out of sight; returns (ID, slot, name) of the player clicked, if any."""
    shown = [c for c in frame.columns if c not in ("ID", "slot")]
    styled = injuries.paint(styled, frame)
    config = {
        c: (st.column_config.TextColumn if c in TEXT_COLUMNS else st.column_config.NumberColumn)(
            width=WIDTHS.get(c, 60)
        )
        for c in shown
    }
    event = st.dataframe(
        styled, hide_index=True, column_order=shown, height=min(len(frame), rows) * ROW_HEIGHT + 38, row_height=ROW_HEIGHT - 1,
        on_select="rerun", selection_mode="single-cell", key=key, column_config=config,
    )  # fmt: skip
    picked = event.selection.cells
    if not picked:
        return None
    row = frame.iloc[int(picked[0][0])]
    return int(row["ID"]), str(row["slot"]), str(row["Player"])


def eligible_view(rec: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": rec["ID"], "slot": rec["Pos"], "Player": rec["Player"], "Pos": rec["Pos"], "Age": rec["Age"],
            "Level": rec["Level"],
            "Tier": rec["Tier"], "Win": rec["Window"], "Value": rec["Value"], "MLB %": rec["MLB"] * 100, "Gain": rec["Gain"],
            "Upside": rec["Upside"], "Ceil": rec["Ceil"], "Floor": rec["Floor"], "Top-50 %": rec["Top50"],
            "Action": rec["Action"],
            "Why": rec["Why"],
        }
    )  # fmt: skip


def style_eligible(frame: pd.DataFrame):
    formats = {c: "{:.1f}" for c in ("Value", "Gain", "Upside", "Ceil", "Floor")} | {
        "MLB %": "{:.0f}",
        "Top-50 %": "{:.0f}",
    }
    style = frame.style.format(formats | {"Age": "{:.0f}"}, na_rep="")
    return style.map(
        lambda v: f"color: {ACTION_COLORS[v]}; font-weight: 700" if v in ACTION_COLORS else "", subset=["Action"]
    )


def removal_view(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    go = np.where(np.arange(len(candidates)) < count, "GO", "")
    return pd.DataFrame(
        {
            "ID": candidates["ID"], "slot": candidates["Pos"], "Go": go, "Player": candidates["Player"], "Pos": candidates["Pos"],
            "Age": candidates["Age"], "Level": candidates["Level"], "Tier": candidates["Tier"], "Value": candidates["Value"],
            "MLB %": candidates["MLB"] * 100, "Keep %": candidates["Keep"] * 100, "Owed $m": candidates["Owed"] / 1e6,
            "Years": candidates["Years"],
        }
    )  # fmt: skip


def style_removal(frame: pd.DataFrame):
    style = frame.style.format(
        {"Value": "{:.1f}", "MLB %": "{:.0f}", "Keep %": "{:.0f}", "Owed $m": "{:.1f}", "Age": "{:.0f}"}, na_rep=""
    )
    return style.apply(lambda row: ["font-weight: 700; color: #d6453d" if row["Go"] else "" for _ in row], axis=1)


def target_view(targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": targets["ID"], "slot": targets["slot"], "Player": targets["Player"], "Pos": targets["slot"],
            "Age": targets["Age"],
            "From": targets["From"], "Value": targets["value"], "Gain": targets["Gain"], "Place": targets["Place"],
        }
    )  # fmt: skip


def style_targets(frame: pd.DataFrame):
    return frame.style.format({"Value": "{:.1f}", "Gain": "{:.1f}", "Age": "{:.0f}"}, na_rep="")


def potential_view(targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": targets["ID"], "slot": targets["slot"], "Player": targets["Player"], "Pos": targets["Pos"],
            "Age": targets["Age"], "From": targets["From"], "Level": targets["Level"], "Tier": targets["Tier"],
            "Win": targets["Window"], "Value": targets["Value"], "Ceil": targets["Ceil"], "Floor": targets["Floor"],
            "Top-50 %": targets["Top50"], "Ceil %": targets["ceiling_pct"] * 100,
        }
    )  # fmt: skip


def style_potential(frame: pd.DataFrame):
    formats = {c: "{:.1f}" for c in ("Value", "Ceil", "Floor")} | {
        "Top-50 %": "{:.0f}",
        "Ceil %": "{:.0f}",
        "Age": "{:.0f}",
    }
    return frame.style.format(formats, na_rep="")


def main() -> None:
    config = load_update_config()
    st.html(org_view.STYLE)
    league_column, first, *links = st.columns([0.9, 1.2, *org_view.LINK_WIDTHS], vertical_alignment="center")
    save = league.choose_save(config, league_column, key="league-rule5")
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
    marks = data_cache.injury_marks(save, season)
    if not marks.empty:
        st.html(injuries.legend_html())
    if "R5" not in frame.columns:
        st.error(
            f"{save} has no Rule 5 flag (R5) in its export: enable it in the game and export the player lists again."
        )
        return
    orgs = sorted(set(acquire.mlb_players(frame)["org"]))
    mine = config[save].get("org")
    org = first.selectbox(
        "Organization", orgs, index=orgs.index(mine) if mine in orgs else 0, label_visibility="collapsed"
    )
    for column, (name, (target, label, icon)) in zip(links, org_view.PAGES.items()):
        try:
            column.page_link(target, label=label, icon=icon, disabled=name == "rule5")
        except StreamlitPageNotFoundError:  # run on its own, without the navigation of viewer/app.py
            break

    settings = rule5.load_settings()
    with st.expander("Rules (defaults from config/rule5.toml; changes here last for this session)"):
        ready = st.slider(
            "Ready for the majors from percentile (value now among the MLB players at his position)",
            0.0,
            0.8,
            settings.ready_percentile,
            0.05,
        )
        potential = st.slider(
            "High potential from ceiling percentile (ceiling among the league's minor leaguers at his position; "
            "a rebuilding team may draft him although he is not ready)",
            0.5,
            1.0,
            settings.potential_percentile,
            0.01,
        )
    settings = replace(settings, ready_percentile=ready, potential_percentile=potential)
    rules = minors.load_rules()
    version = scales.version_of(get_save(save).ratings)
    table = rule5.player_table(frame, org, settings, rules, version)
    rec = rule5.recommend(table, settings)
    starters = acquire.team_starters(frame, acquire.Settings())
    exact = table.attrs.get("exact", True)

    st.caption(
        " ".join(
            [
                f"**{org}** {save} {season}: draft eligible players of your organization (the R5 flag of the export).",
                "Protect = a prospect with development left whose ceiling would start for your MLB team and who is ready for the",
                "majors, a top prospect, a prospect with high potential (a rebuilding team drafts on potential and does not mind",
                "the roster spot), or a player who would start now.",
                "Low risk = a prospect who is not ready and has no such potential (a team that drafts him must keep him on its",
                "26-man roster all season).",
                "Expose = the rest. The 40-man roster counts the players on it today; a free agent you sign needs a slot too.",
                "Click a player for his ratings.",
            ]
        )
    )
    if not exact:
        st.warning(
            f"The export has no 40-man flag ({settings.column}): the roster is estimated from the MLB level and the secondary "
            "roster time, so the count can be off. Add the flag to the player list export for an exact count."
        )
    controls = st.columns([1, 2])
    signings = controls[0].number_input(
        "Free agents I expect to sign", 0, 15, 0, help="Each one takes a 40-man slot too"
    )
    options = list(rec["Player"] + " (" + rec["Pos"] + ", " + rec["Level"] + ")")
    default = list(
        rec.loc[rec["Action"] == rule5.PROTECT, "Player"]
        + " ("
        + rec.loc[rec["Action"] == rule5.PROTECT, "Pos"]
        + ", "
        + rec.loc[rec["Action"] == rule5.PROTECT, "Level"]
        + ")"
    )
    protect = controls[1].multiselect("Players to protect", options, default, key=f"protect-{save}-{org}-{ready}")
    room = rule5.roster_room(table, settings, len(protect), int(signings))
    metrics = st.columns(5)
    metrics[0].metric("On the 40-man" + ("" if exact else " (estimate)"), f"{room['on_forty']} of {settings.size}")
    metrics[1].metric("Open slots", room["open"])
    metrics[2].metric("Draft eligible", len(rec))
    metrics[3].metric("Slots the plans need", room["needed"])
    metrics[4].metric("Players to take off", room["to_clear"])

    generation = st.session_state.setdefault("rule5_generation", 0)  # a new key clears the row selection of every table
    picked = None
    st.subheader("Your players who can be drafted")
    if rec.empty:
        st.write("None of your players is draft eligible.")
    else:
        view = injuries.add(eligible_view(rec), marks)
        picked = show(view, style_eligible(view), f"rule5-eligible-{save}-{org}-{generation}", rows=20)

    st.subheader("Who could come off the 40-man")
    candidates = rule5.removal_candidates(table, starters, org)
    if room["to_clear"] > 0:
        st.write(
            f"{room['to_clear']} players have to come off to fit {room['needed']} in {room['open']} open slots "
            "(the first ones below)."
        )
    else:
        st.write(
            "Nobody has to come off: the open slots cover the plans. The ones with the least to keep them are listed anyway."
        )
    if candidates.empty:
        st.write("No player on the 40-man who is not a starter.")
    else:
        view = injuries.add(removal_view(candidates, room["to_clear"]), marks)
        picked = picked or show(view, style_removal(view), f"rule5-remove-{save}-{org}-{generation}")

    st.subheader("Players to draft from other organizations")
    targets = rule5.draft_targets(frame, org, settings, acquire.load_settings())
    if targets.empty:
        st.write("No draft eligible player of another organization would improve your MLB roster.")
    else:
        view = injuries.add(target_view(targets.head(30)), marks)
        picked = picked or show(view, style_targets(view), f"rule5-targets-{save}-{org}-{generation}")
        st.caption(
            "Gain = runs over the starter of the place (or over your best other player when it is empty). "
            "A pick takes a 40-man slot."
        )

    st.subheader("High-potential prospects to draft")
    potential = rule5.potential_targets(frame, org, settings, rules, version)
    if potential.empty:
        st.write("No draft eligible prospect of another organization is exposed with that much potential.")
    else:
        view = injuries.add(potential_view(potential.head(30)), marks)
        picked = picked or show(view, style_potential(view), f"rule5-potential-{save}-{org}-{generation}")
        st.caption(
            "Exposed draft eligible prospects (not on their 40-man roster) whose ceiling is in the best of the league at their "
            "position, whether or not they help the MLB roster now: for a weak season you could take one and keep him on the "
            "26-man roster for his potential. Ceil % = where the ceiling stands among the minor leaguers of his kind."
        )
    if picked:
        st.session_state["rule5_generation"] = generation + 1
        player_id, slot, name = picked
        org_view.player_dialog(save, season, player_id, slot, name, f"{org}, {slot}")


main()
