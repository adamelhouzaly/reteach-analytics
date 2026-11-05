"""Reteach — Student Misconception Analytics (Streamlit entry point).

Run:  streamlit run app.py

This file only sets up the page, shared state and navigation. Each page lives
in ``views/``; shared presentation helpers live in ``ui/``; all calculations
live in the ``reteach`` package.
"""

from __future__ import annotations

import os

import streamlit as st

from ui import state, theme

st.set_page_config(page_title="Reteach — Misconception Analytics", layout="wide",
                   initial_sidebar_state="collapsed")
theme.inject_css()
state.init()

PAGES = {
    "overview": st.Page("views/overview.py", title="Overview", default=True),
    "misconceptions": st.Page("views/misconceptions.py", title="Misconceptions",
                              url_path="misconceptions"),
    "groups": st.Page("views/groups.py", title="Student groups", url_path="groups"),
    "review": st.Page("views/review.py", title="Review answers", url_path="review"),
    "upload": st.Page("views/upload.py", title="Upload data", url_path="upload"),
    "evaluation": st.Page("views/evaluation.py", title="Model evaluation",
                          url_path="evaluation"),
}
st.session_state["_pages"] = PAGES  # lets views link to each other with st.switch_page

page = st.navigation(list(PAGES.values()), position="top")
try:
    page.run()
except Exception as exc:  # noqa: BLE001 - deliberate UI guard: never show a traceback
    # (st.switch_page / st.rerun use BaseException subclasses, so they pass through.)
    if os.environ.get("RETEACH_DEBUG"):
        raise
    st.error("Something went wrong while showing this page. Try **Upload data → Reset to "
             f"demo data**, or check your uploaded files. ({type(exc).__name__})")
