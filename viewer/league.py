"""The league (save) a viewer page shows: one choice shared by all pages, kept for the browser session."""

import streamlit as st

SESSION_KEY = "viewer_save"


def choose_save(config: dict, container=None, key: str = "league") -> str:
    """A selectbox of the saves of config/update.toml (the first is the default); the choice follows to the other pages."""
    saves = list(config)
    current = st.session_state.get(SESSION_KEY)
    index = saves.index(current) if current in saves else 0
    chosen = (container or st).selectbox("League", saves, index=index, key=key, label_visibility="collapsed")
    st.session_state[SESSION_KEY] = chosen
    return chosen


PARK_KEY = "viewer_park"


def park_toggle(container, team: str | None, key: str, available: bool = True) -> bool:
    """A toggle for park-adjusted values (the values as if half the player's games were in the park of `team`); the choice
    follows to the other pages. Off and greyed out when there is no park to adjust to (no team, or the league has no park
    data)."""
    usable = available and team is not None
    label = f"Park-adjusted values ({team} park)" if usable else "Park-adjusted values"
    help_text = (
        "Values as if half the player's games were played in this park: home runs, doubles, triples and hits move with the park "
        "factors, a pitcher's runs allowed with its run factor. Fielding and baserunning are unchanged."
        if usable
        else "Needs an organization (not all organizations or free agents) and a league with park factors and park models."
    )
    chosen = container.toggle(
        label,
        value=bool(st.session_state.get(PARK_KEY, False)) and usable,
        key=key,
        disabled=not usable,
        help=help_text,
    )
    if usable:
        st.session_state[PARK_KEY] = chosen
    return chosen and usable


@st.cache_data(show_spinner=False)
def park_context(save: str, season: int):
    """The park context of a league (viewer/parks.py), None when it has no park data."""
    from viewer import parks

    return parks.load_context(save, season)


def park_available(save: str, season: int) -> bool:
    return park_context(save, season) is not None


def park_frame(save: str, season: int, frame, team: str | None, on: bool, own: bool = False):
    """`frame` valued in the park of `team` when the toggle is on (and the league has park data), else as it is."""
    from viewer import parks

    ctx = park_context(save, season)
    if not on or ctx is None or (team is None and not own):
        return frame
    return parks.adjust(frame, ctx, team=team, own=own)
