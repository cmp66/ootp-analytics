"""The explorer screen: any list, many options (plan 15).

The sidebar picks the save, season, list and filters; the main area shows the ordered list and a table per position and offers
the Excel report. All query logic is in viewer/queries.py and viewer/report.py, so this file only holds widgets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from scripts.update_all import load_update_config  # noqa: E402
from util.seasons import available_seasons  # noqa: E402
from viewer import league, queries, report  # noqa: E402

METRICS = {
    "p_top10": "Chance of top 10",
    "p_11_50": "Chance of 11th to 50th",
    "p_top50": "Chance of top 50",
    "rank_score": "Blend of floor / median / ceiling",
    "ceiling": "Ceiling (90th percentile)",
    "floor": "Floor (10th percentile)",
    "value": "Median value",
    "season_value": "Value after missed time",
    "p_start": "Chance to start for my team",
    "p_roster": "Chance to make my roster",
}
POOL_LABELS = {
    "all": "Everyone", "mlb": "MLB players", "org": "My organization", "others": "Other organizations (trade targets)",
    "free_agents": "Free agents", "iafa": "IAFA prospects", "draft": "Draft class",
}  # fmt: skip
HORIZON_LABELS = {
    1: "1 season", 2: "2 seasons", 3: "3 seasons", 5: "5 seasons", 7: "7 seasons", 10: "10 seasons", 99: "career (best of 10)",
}  # fmt: skip
COLUMN_FORMATS = {
    "key": "%.3f", "value_now": "%.1f", "bat_value": "%.1f", "run_value": "%.1f", "field_value": "%.1f",
    "pit_value": "%.1f", "floor": "%.1f", "value": "%.1f", "ceiling": "%.1f", "season_value": "%.1f",
    "injury_cost": "%.1f", "p_top10": "%.2f", "p_top50": "%.2f", "p_stay": "%.2f", "p_start": "%.2f", "p_roster": "%.2f",
}  # fmt: skip


@st.cache_data(show_spinner="Loading the player files...")
def load(save: str, season: int) -> pd.DataFrame:
    return queries.load_outlook(save, season)


def show(frame: pd.DataFrame) -> None:
    config_columns = {c: st.column_config.NumberColumn(format=f) for c, f in COLUMN_FORMATS.items() if c in frame}
    st.dataframe(frame, hide_index=True, width="stretch", column_config=config_columns)


def filter_lines(
    save, season, pool_name, metric, horizon, slots, ages, levels, include_depth, org
) -> list[tuple[str, str]]:
    """The filters as text, printed above every table of the report."""
    return [
        ("Save", f"{save} {season}"),
        ("Players from", POOL_LABELS[pool_name]),
        ("Sorted by", f"{METRICS[metric]}, {HORIZON_LABELS[horizon]}"),
        ("Positions", ", ".join(slots) if slots else "all (each player once, at his best position)"),
        ("Age", f"{ages[0]} to {ages[1]}"),
        ("Levels", ", ".join(levels) if levels else "all"),
        ("Roster filler", "included" if include_depth else "left out"),
        ("My organization", org or "(none)"),
    ]


def main() -> None:
    config = load_update_config()
    defaults, specs = queries.load_list_specs()

    try:
        st.sidebar.page_link("screens/org_overview.py", label="Organization overview", icon=":material/arrow_back:")
    except StreamlitPageNotFoundError:  # run on its own, without the navigation of viewer/app.py
        pass
    st.sidebar.title("Player lists")
    save = league.choose_save(config, st.sidebar, key="league-explorer")
    seasons = available_seasons(save)
    if not seasons:
        st.error(f"No seasons with data for {save}.")
        return
    season = st.sidebar.selectbox("Season", sorted(seasons, reverse=True))
    try:
        frame = load(save, season)
    except FileNotFoundError:
        st.error(f"{save} {season} has no outlook files yet: run `python scripts/update_all.py {save}`.")
        return

    name = st.sidebar.selectbox("List", list(specs), format_func=lambda n: specs[n].title)
    spec = specs[name]
    horizons = queries.available_horizons(frame)
    first = (
        horizons.index(defaults["horizon"]) if defaults.get("horizon") in horizons else 0
    )  # the default may be a dropped one
    horizon = st.sidebar.selectbox("Seasons ahead", horizons, index=first, format_func=HORIZON_LABELS.get)
    metric = st.sidebar.selectbox(
        "Sort by", list(METRICS), index=list(METRICS).index(spec.metric), format_func=METRICS.get
    )

    orgs = sorted(o for o in frame["org"].dropna().unique() if o not in ("FA", "nan", ""))
    org_default = config[save].get("org")
    org = st.sidebar.selectbox(
        "My organization",
        [None] + orgs,
        format_func=lambda o: "(none)" if o is None else o,
        index=orgs.index(org_default) + 1 if org_default in orgs else 0,
    )

    park_on = league.park_toggle(st.sidebar, org, "park-explorer", league.park_available(save, season))
    frame = league.park_frame(save, season, frame, org, park_on)  # values in the park of my organization

    pools = ["all", "mlb", "org", "others", "free_agents", "iafa", "draft"]
    pool_name = st.sidebar.selectbox("Players from", pools, index=pools.index(spec.pool), format_func=POOL_LABELS.get)
    slots = st.sidebar.multiselect("Positions", queries.SLOT_ORDER, default=list(spec.slots))
    age_min, age_max = st.sidebar.slider("Age", 15, 45, (spec.age_min or 15, spec.age_max or 45))
    levels = st.sidebar.multiselect("Levels", sorted(frame["level_name"].unique()), default=list(spec.levels))
    include_depth = st.sidebar.checkbox("Include roster filler", value=spec.include_depth)
    top = st.sidebar.number_input("Rows in the ordered list", 10, 500, defaults.get("top", 50), step=10)
    per_position = st.sidebar.number_input("Rows per position", 5, 200, 25, step=5)

    needs_org = metric in ("p_start", "p_roster") or pool_name in ("org", "others")
    st.header(spec.title)
    st.caption(
        f"{save} {season}, {HORIZON_LABELS[horizon]}. Each player appears once, at his best "
        + ("of the positions chosen." if slots else "position; choose positions to see him at those.")
    )
    if needs_org and org is None:
        st.info("This list compares with your roster: pick your organization in the sidebar.")
        return

    pool = queries.select_pool(frame, pool_name, org)
    pool = queries.apply_filters(pool, slots, age_min, age_max, levels, include_depth)
    if org is not None:  # with an organization every list also shows the chance to start / make its roster
        pool = queries.fit_table(pool, org, horizon, thresholds_from=frame)
    report_spec = report.ReportSpec(
        title=f"{spec.title} - {save} {season}",
        metric=metric,
        horizon=horizon,
        filters=filter_lines(
            save, season, pool_name, metric, horizon, slots, (age_min, age_max), levels, include_depth, org
        ),
        top=int(top),
        top_per_position=int(per_position),
    )
    ordered = report.ordered_list(pool, report_spec)
    if ordered.empty:
        st.warning(
            "No players match. In an export the draft class and the IAFA prospects are only present before the draft."
        )
        return
    blocks = report.position_blocks(pool, report_spec)
    lowest = report.ordered_list(pool, report_spec, lowest=True)
    with st.expander("What the filters select", expanded=False):
        for label, text in report_spec.filters:
            st.write(f"**{label}:** {text}")
    list_tab, position_tab = st.tabs(["Ordered list", "By position"])
    with list_tab:
        show(ordered)
        st.download_button(
            "Download csv", ordered.to_csv(index=False), file_name=f"{save}-{season}-{name}.csv", mime="text/csv"
        )
    with position_tab:
        for slot in blocks:
            st.subheader(f"{slot} ({len(pool[pool['slot'] == slot])} players)")
            show(blocks[slot])
    st.download_button(
        "Download Excel report",
        report.write_excel(ordered, lowest, blocks, report_spec),
        file_name=f"{save}-{season}-{name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


main()
