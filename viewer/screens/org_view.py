"""Shared page of the organization overview and the minor league overview (plan 15): one organization, one screen.

Choices: the organization (default: `org` of the first save in config/update.toml)
; on the minor league page also floor / ceiling. Click a column header of one of the three
long tables to sort all of the organization's players by it; click a player to see his ratings.
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from scripts.update_all import load_update_config  # noqa: E402
from util import paths  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from viewer import league, overview, player_card, queries  # noqa: E402

ROW_HEIGHT = 21  # pixels of a table row; lower it if the screen does not fit
HEADER_HEIGHT = 36
SHOWN_ROWS = {"Position players": 20, "Starting pitchers": 20, "Relief pitchers": 20}
POSITION_ROWS = 10
KINDS = {"Position players": "hit", "Starting pitchers": "sp", "Relief pitchers": "rp"}
COLORS = {"hit": "#1f5f99", "pos": "#3b7a57", "sp": "#a0522d", "rp": "#7a4a99"}
RUN_COLUMNS = ["Off", "Def", "Run"]
NUMBER_FORMATS = {"Value": "{:.1f}", "Ceiling": "{:.1f}", "Floor": "{:.1f}", "Top-10 %": "{:.0f}", "Top-50 %": "{:.0f}"}

EXPLANATION = {
    "five": {
        "ceiling": "**{who}** {save} {season}: Ceiling = the best value he can reach (90th percentile) within "
        "5 seasons; Top-10 % = his best chance of being a top-10 player at the position in one of those seasons. Click a "
        "header to sort the whole system.",
        "floor": "**{who}** {save} {season}: Floor = the best value he is likely to at least reach (10th "
        "percentile) within 5 seasons; Top-50 % = his best chance of being a top-50 player at the position in one of those "
        "seasons. Click a header to sort.",
    },
    "career": {
        "ceiling": "**{who}** {save} {season}: Ceiling = the 90th percentile of his best season within the next "
        "10; Top-10 % = his chance of being a top-10 player at the position at some point in those 10 seasons.",
        "floor": "**{who}** {save} {season}: Floor = the 10th percentile of his best season within the next 10; "
        "Top-50 % = his chance of being a top-50 player at the position at some point in those 10 seasons.",
    },
}

STYLE = """
<style>
.block-container {padding: 0.5rem 1rem 0 1rem; max-width: 100%;}
header[data-testid="stHeader"] {display: none;}
[data-testid="stVerticalBlock"] {gap: 0.3rem;}
.bar {margin: 0; padding: 2px 8px; font-size: 14px; font-weight: 600; color: #fff; border-radius: 6px 6px 0 0;}
</style>
"""


def run_color(value) -> str:
    if pd.isna(value):
        return ""
    return "color: #2e9e57" if value > 0.05 else "color: #d6453d" if value < -0.05 else ""


def styled(frame: pd.DataFrame):
    formats = {c: "{:.1f}" for c in RUN_COLUMNS if c in frame} | {c: f for c, f in NUMBER_FORMATS.items() if c in frame}
    style = frame.style.format(formats, na_rep="")
    style = style.map(run_color, subset=[c for c in RUN_COLUMNS if c in frame])
    return style.apply(lambda row: ["font-weight: 700" if row["Lev"] == "MLB" else "" for _ in row], axis=1)


COLUMN_WIDTHS = {  # pixels: the three narrow cards must fit their columns without clipping or a sideways scrollbar
    "Player": 112, "Pos": 40, "Age": 40, "Lev": 46, "Off": 50, "Def": 50, "Run": 50, "Value": 56, "Ceiling": 62,
    "Floor": 56, "Top-10 %": 66, "Top-50 %": 66, "Org": 42,
}  # fmt: skip


def column_config(columns: list[str]) -> dict:
    config = {}
    for column in columns:
        kind = (
            st.column_config.TextColumn if column in ("Player", "Pos", "Lev", "Org") else st.column_config.NumberColumn
        )
        config[column] = kind(width=COLUMN_WIDTHS.get(column, 50))
    return config


def position_filter(key: str, slots: list[str]) -> list[str]:
    """A popover of checkboxes, one per position (none checked = all positions); returns the checked positions."""
    chosen = [slot for slot in slots if st.session_state.get(f"{key}-{slot}", False)]
    label = f"Pos: {', '.join(chosen)}" if chosen else "Pos \u25be"
    with st.popover(label, help="Filter by position: check one or more", use_container_width=True):
        for slot in slots:
            st.checkbox(slot, key=f"{key}-{slot}")
    return [slot for slot in slots if st.session_state.get(f"{key}-{slot}", False)]


def show(
    title: str, frame: pd.DataFrame, kind: str, rows: int, key: str, positions: list[str] | None = None, rebuild=None
) -> int | None:
    """A titled table; returns the player ID of the row clicked, if any. Header clicks sort all of `frame`.

    With `positions` (the position choices of a table that has a Pos column) the title row also holds a checkbox menu that
    limits the table to the checked positions: `rebuild(chosen)` returns the table for them, a player at his best of them.
    """
    bar = f'<div class="bar" style="background:{COLORS[kind]}">{html.escape(title)}</div>'
    if positions:
        title_column, menu = st.columns([3, 1], vertical_alignment="center", gap="small")
        title_column.html(bar)
        with menu:
            chosen = position_filter(f"pos-{key.split('-')[0]}", positions)
        if chosen:
            frame = rebuild(chosen)
    else:
        st.html(bar)
    shown = [c for c in frame.columns if c not in ("ID", "Slot")]
    event = st.dataframe(
        styled(frame),
        hide_index=True,
        column_order=shown,
        height=min(len(frame), rows) * ROW_HEIGHT + HEADER_HEIGHT,
        row_height=ROW_HEIGHT,
        on_select="rerun",
        selection_mode="single-cell",  # a cell click, unlike a row click, adds no checkbox column,
        key=key,
        column_config=column_config(shown),
    )
    picked = event.selection.cells
    if not picked:
        return None
    position = int(picked[0][0])
    return int(frame["ID"].iloc[position]), str(frame["Slot"].iloc[position])


@st.cache_data(show_spinner=False)
def load_ratings(save: str, season: int) -> pd.DataFrame:
    return (
        pd.read_csv(paths.output_file(save, season, "player-data"), low_memory=False)
        .drop_duplicates("ID")
        .set_index("ID")
    )


@st.dialog("Player ratings", width="large", on_dismiss="rerun")
def player_dialog(save: str, season: int, player_id: int, slot: str, name: str, extra: str) -> None:
    row = load_ratings(save, season).loc[player_id]
    st.markdown(f"### {player_card.full_name(row, name)}")
    st.html(player_card.card_html(row, slot, f"{extra} - ratings on the 20-80 scale"))


PAGES = {  # page key -> (file, link label, icon)
    "org": ("screens/org_overview.py", "Organization", ":material/home:"),
    "minors": ("screens/minors_overview.py", "Minor leagues", ":material/school:"),
    "draft": ("screens/draft_overview.py", "MLB draft", ":material/format_list_numbered:"),
    "iafa": ("screens/iafa_overview.py", "IAFA", ":material/public:"),
    "depth": ("screens/depth_chart.py", "Depth chart", ":material/stadium:"),
    "management": ("screens/minor_management.py", "Level moves", ":material/swap_vert:"),
    "explorer": ("screens/explorer.py", "Explorer", ":material/search:"),
}
POOL_TITLES = {"draft": "MLB draft class", "iafa": "International amateurs (IAFA)"}
POOL_NAMES = {"minors": "org", "draft": "draft", "iafa": "iafa"}
ALL_ORGS = "All organizations"
FREE_AGENTS = "Free agents"


def render(page: str = "org") -> None:
    """Draw a page: "org" (current value of one organization), "minors" (its players not on the MLB roster) or "draft" / "iafa"
    (players of no organization yet). All but the first show floor / ceiling instead of the current value."""
    config = load_update_config()
    st.html(STYLE)
    league_column, first, *links = st.columns([0.9, 1.2, 1, 1.3, 1, 0.8, 1, 1.1, 1.3], vertical_alignment="center")
    save = league.choose_save(config, league_column, key=f"league-{page}")
    seasons = available_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = seasons[-1]

    @st.cache_data(show_spinner="Loading the player files...")
    def load(save_name: str, season_number: int):
        return queries.load_outlook(save_name, season_number)

    try:
        frame = load(save, season)
    except FileNotFoundError:
        st.error(f"{save} {season} has no outlook files yet: run `python scripts/update_all.py {save}`.")
        return
    has_org = page in ("org", "minors")
    projection = page != "org"
    orgs = overview.organizations(frame)
    mine = config[save].get("org")
    if has_org:
        choices = [ALL_ORGS, *([FREE_AGENTS] if page == "org" else []), *orgs]
        org = first.selectbox(
            "Organization", choices, index=choices.index(mine) if mine in choices else 0, label_visibility="collapsed"
        )
    else:
        org = None
        first.markdown(f"**{POOL_TITLES[page]}**")
    for column, (key, (target, label, icon)) in zip(links, PAGES.items()):
        try:
            column.page_link(target, label=label, icon=icon, disabled=key == page)
        except StreamlitPageNotFoundError:  # run on its own, without the navigation of viewer/app.py
            break
    mode, window = "current", "five"
    if projection:
        choice, span, caption = st.columns([1.2, 2.4, 6], vertical_alignment="center")
        mode = choice.radio("Projection", ["Ceiling", "Floor"], horizontal=True, label_visibility="collapsed").lower()
        windows = {"Next 5 seasons": "five", "Career (best of 10 seasons)": "career"}
        if "ceiling_h99" not in frame.columns:
            windows = {"Next 5 seasons": "five"}  # outlook files made before the career view
        window = windows[span.radio("Window", list(windows), horizontal=True, label_visibility="collapsed")]
        who = (
            (f"{org} minor leaguers" if org != ALL_ORGS else "Minor leaguers of all organizations")
            if has_org
            else POOL_TITLES[page]
        )
        caption.caption(EXPLANATION[window][mode].format(who=who, save=save, season=season))
    else:
        st.markdown(
            f"**{org}** {save} {season}: Value = the player's current value in runs above the league's average starter at the "
            "position (Off + Def + Run). Click a header to sort everybody listed, a player for his ratings."
        )
    pool_name = {ALL_ORGS: "all_orgs", FREE_AGENTS: "free_agents"}.get(org, POOL_NAMES.get(page, "org"))
    owner, with_all = None if org in (ALL_ORGS, FREE_AGENTS) else org, org == ALL_ORGS
    tables = overview.pool_overview(
        frame, pool_name, owner, mode, minors_only=page == "minors", window=window, with_org=with_all
    )
    generation = st.session_state.setdefault("generation", 0)  # a new key clears the row selection of every table
    picked = None

    def table(title, kind, rows, column=None, positions=None):
        def rebuild(chosen):
            return overview.pool_overview(frame, pool_name, owner, mode, page == "minors", window, with_all, chosen)[
                title
            ]

        nonlocal picked
        with column if column is not None else st.container():
            hit = show(
                title,
                tables[title],
                kind,
                rows,
                f"{page}-{org}-{mode}-{window}-{title}-{generation}",
                positions,
                rebuild,
            )
        picked = picked or (hit and (title, *hit))

    top = st.columns([2, 1, 1], gap="small")
    for column, title in zip(top, SHOWN_ROWS):
        table(
            title,
            KINDS[title],
            SHOWN_ROWS[title],
            column,
            overview.HITTER_SLOTS if title == "Position players" else None,
        )
    slots = overview.HITTER_SLOTS
    per_row = (len(slots) + 1) // 2
    for chunk in (slots[:per_row], slots[per_row:]):
        for column, slot in zip(st.columns(per_row, gap="small"), chunk):
            table(slot, "pos", POSITION_ROWS, column)

    if picked:
        title, player_id, slot = picked
        row = tables[title].loc[tables[title]["ID"] == player_id].iloc[0]
        st.session_state["generation"] = generation + 1
        player_dialog(
            save,
            season,
            player_id,
            slot,
            row["Player"],
            f"{org or POOL_TITLES[page]}, age {row['Age']}, {row['Lev']}, {slot}",
        )
