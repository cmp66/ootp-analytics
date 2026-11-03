"""Entry point of the viewer (plan 15). Start it with scripts/run_viewer.py.

Screens: the organization overview (focused, one screen) and the explorer (any list, many options).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

SCREENS = Path(__file__).resolve().parent / "screens"

st.set_page_config(page_title="OOTP", layout="wide", initial_sidebar_state="collapsed")
navigation = st.navigation(
    [
        st.Page(SCREENS / "org_overview.py", title="Organization overview", url_path="overview", default=True),
        st.Page(SCREENS / "minors_overview.py", title="Minor leagues", url_path="minors"),
        st.Page(SCREENS / "draft_overview.py", title="MLB draft", url_path="draft"),
        st.Page(SCREENS / "iafa_overview.py", title="IAFA", url_path="iafa"),
        st.Page(SCREENS / "explorer.py", title="Explorer", url_path="explorer"),
    ],
    position="top",
)
navigation.run()
