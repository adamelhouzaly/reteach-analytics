"""Upload data: templates, validation feedback, and switching datasets."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reteach import validation as val
from ui import components as ui
from ui import state

DATA = state.DATA

ui.page_header("Upload data", "Use your own quiz. Files are only read as CSV text — never run.")

# --- Current data -------------------------------------------------------------
ds = state.dataset()
with ui.card("current-data"):
    a, b = st.columns([3, 1], vertical_alignment="center")
    with a:
        st.markdown("**Data in use**")
        n_students = ds["responses"]["student_id"].nunique()
        n_questions = ds["questions"]["question_id"].nunique()
        ui.muted(f"{ds['label']} · {n_students} students · {n_questions} questions"
                 + (" · fictional demo data" if ds["synthetic"] else ""))
    with b:
        if st.button("Reset to demo data", key="reset_demo", icon=":material/restart_alt:",
                     help="Reload the demo class and clear this session's confirmed labels. "
                          "Saved label files are kept."):
            state.reset_to_demo()
            state.flash("Demo data restored.")
            st.rerun()

st.space("small")

# --- Step 1: templates ---------------------------------------------------------
with ui.card("step-templates"):
    ui.eyebrow("Step 1")
    st.markdown("**Download a template**")
    with st.container(horizontal=True, gap="small"):
        st.download_button("Questions template",
                           (DATA / "templates" / "questions_template.csv").read_bytes(),
                           "questions_template.csv", "text/csv", icon=":material/download:")
        st.download_button("Responses template",
                           (DATA / "templates" / "responses_template.csv").read_bytes(),
                           "responses_template.csv", "text/csv", icon=":material/download:")
        st.download_button("Demo questions",
                           (DATA / "demo" / "questions.csv").read_bytes(),
                           "demo_questions.csv", "text/csv", icon=":material/download:")
        st.download_button("Messy sample (to test validation)",
                           (DATA / "invalid" / "responses_messy.csv").read_bytes(),
                           "responses_messy.csv", "text/csv", icon=":material/download:",
                           type="tertiary")
    with st.expander("What goes in each file"):
        for name, cols in val.SCHEMA_DOCS.items():
            st.markdown(f"**{name}.csv**")
            st.dataframe(pd.DataFrame(cols.items(), columns=["Column", "What to put there"]),
                         hide_index=True, width="stretch",
                         column_config={"What to put there":
                                        st.column_config.TextColumn(width="large")})
        st.markdown("**Skipped and missing answers.** A blank `selected_option` counts as "
                    "*unanswered*; a student with no row for a question is *missing*. Neither "
                    "counts toward accuracy — both appear in response coverage. If a student "
                    "has two rows for the same question, the last one is used.")

st.space("small")

# --- Step 2: upload + validate ---------------------------------------------------
with ui.card("step-upload"):
    ui.eyebrow("Step 2")
    st.markdown("**Upload and check your files**")
    q_col, r_col = st.columns(2)
    with q_col:
        qfile = st.file_uploader("Questions (one row per answer option)", type=["csv"],
                                 key="q_upload")
        use_demo_q = st.checkbox("Use the demo questions instead", value=qfile is None,
                                 key="use_demo_q")
    with r_col:
        rfile = st.file_uploader("Student responses", type=["csv"], key="r_upload")

    def show(name: str, res: val.ValidationResult) -> None:
        if res.errors:
            st.error(f"**{name}: fix {len(res.errors)} problem"
                     f"{'s' if len(res.errors) > 1 else ''} and upload again.**\n\n"
                     + "\n".join(f"- {e}" for e in res.errors), icon=":material/error:")
        if res.warnings:
            st.warning(f"**{name}: {len(res.warnings)} note"
                       f"{'s' if len(res.warnings) > 1 else ''} — the rest of the file can "
                       "still be used.**\n\n" + "\n".join(f"- {w}" for w in res.warnings),
                       icon=":material/warning:")
        if res.ok and not res.warnings:
            st.success(f"{name}: no problems found.", icon=":material/check_circle:")

    if rfile is None:
        st.caption("Upload a responses file to check it.")
    if st.button("Check files", type="primary", disabled=rfile is None, key="validate"):
        qres = (val.validate_questions(DATA / "demo" / "questions.csv")
                if use_demo_q or qfile is None else val.validate_questions(qfile.getvalue()))
        show("Questions", qres)
        if qres.ok:
            rres = val.validate_responses(rfile.getvalue(), qres.df)
            show("Responses", rres)
            if rres.ok:
                st.session_state.pending = {
                    "questions": qres.df, "responses": rres.df,
                    "label": f"Uploaded · {rfile.name}", "synthetic": False,
                    "warnings": qres.warnings + rres.warnings}
        else:
            st.session_state.pop("pending", None)

    pending = st.session_state.get("pending")
    if pending:
        n = pending["responses"]["student_id"].nunique()
        st.markdown(f"Ready to use: **{n} students**, {len(pending['responses'])} answers.")
        if st.button("Use this data", type="primary", key="use_upload",
                     icon=":material/check:"):
            state.use_dataset(st.session_state.pop("pending"))
            state.flash("Your uploaded data is now in use on every page.")
            ui.go("overview")
