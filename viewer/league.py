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
