"""Misconceptions: one evidence card per teacher-defined misconception."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reteach import taxonomy as tx
from ui import components as ui
from ui import state
from ui import theme as th

ui.page_header("Misconceptions",
               "Patterns found from the answer key: each wrong option is linked to the "
               "misconception it usually signals. No machine learning is used here.")
ms = ui.min_support_control("mis")
ctx = state.context()
mis = ctx.misconceptions
n = ctx.cov.n_students

# Focused misconception (from Overview) first, then by recurring students.
focus = st.session_state.pop("focus_misconception", None)
mis = mis.assign(is_focus=mis["misconception"] == focus).sort_values(
    ["is_focus", "students_recurring", "selections"], ascending=False)

# --- Comparison chart -----------------------------------------------------------
with ui.card("mis-chart"):
    st.markdown("### Students showing each pattern")
    labels = [th.wrap_label(tx.title(m), 26) for m in mis["misconception"]]
    fig = go.Figure()
    fig.add_bar(name=f"Recurring ({ms}+ questions)", y=labels, x=mis["students_recurring"],
                orientation="h", marker=dict(color=th.TEAL, cornerradius=4),
                text=mis["students_recurring"], textposition="outside", cliponaxis=False,
                hovertemplate="%{x} students — recurring<extra></extra>")
    fig.add_bar(name="At least once", y=labels, x=mis["students_any"], orientation="h",
                marker=dict(color=th.TEAL_LIGHT, cornerradius=4), text=mis["students_any"],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{x} students — at least once<extra></extra>")
    th.style_bar_chart(fig, 90 + 74 * len(mis), x_title=f"students (of {n})",
                       x_max=max(1, int(mis["students_any"].max())) * 1.15)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    th.legend_top(fig)
    st.plotly_chart(fig, config=th.PLOTLY_CONFIG, width="stretch")

st.space("small")

# --- One card per misconception -------------------------------------------------
ev_all = ctx.evidence
scored = ctx.scored
for r in mis.itertuples():
    label = r.misconception
    cat = tx.CATEGORIES[label]
    with ui.card(f"mis-{label}", variant="priority" if r.is_focus else ""):
        if r.is_focus:
            ui.eyebrow("Selected from Overview")
        st.markdown(f"### {cat.title}")
        ui.muted(cat.summary)
        st.space("small")

        s1, s2, s3 = st.columns(3)
        with s1:
            ui.stat(f"{r.students_any} of {r.students_with_opportunity}",
                    "students showed it at least once",
                    "Denominator: students who answered at least one question offering this "
                    "wrong option.")
        with s2:
            ui.stat(r.students_recurring, f"students meet the {ms}-question threshold",
                    "Chose a matching wrong option on at least this many different questions.")
        with s3:
            ui.stat(f"{th.pct(r.selection_rate)}",
                    f"of chances ({r.selections} of {r.opportunities} answers)",
                    "Answers choosing this wrong option, out of all answers to questions "
                    "that offered it.")

        if ms > r.n_supporting_questions:
            st.info(f"Only {th.plural(r.n_supporting_questions, 'question')} in this quiz "
                    f"test this pattern, so no student can reach the {ms}-question threshold.",
                    icon=":material/info:")

        st.markdown(f"**Try this:** {cat.activity}")

        ev = ev_all[(ev_all["misconception"] == label) & (ev_all["n_supporting"] > 0)]
        ev = ev.sort_values(["n_supporting", "student_id"], ascending=[False, True])
        with st.expander(f"Students ({len(ev)})"):
            if ev.empty:
                st.caption("No student chose this wrong option.")
            else:
                st.dataframe(pd.DataFrame({
                    "Student": ev["student_id"],
                    "Status": ["Recurring" if k >= ms else "Not enough evidence yet"
                               for k in ev["n_supporting"]],
                    "Questions": [f"{k} of {o}" for k, o in zip(ev["n_supporting"],
                                                                ev["opportunities"])],
                    "Which questions": ev["supporting_questions"].str.join(", "),
                }), hide_index=True, width="stretch", height=min(36 * len(ev) + 40,
                                                                           300))
        sup = ctx.questions[ctx.questions["misconception_label"] == label]
        with st.expander(f"Supporting questions ({len(sup)})"):
            st.dataframe(pd.DataFrame({
                "Question": sup["question_id"],
                "Wrong option": sup["option_id"] + " · " + sup["option_text"],
                "Prompt": sup["prompt"],
            }), hide_index=True, width="stretch")
        examples = scored[(scored["misconception_label"] == label)
                          & (scored["explanation"] != "")]
        with st.expander("Student examples and worked example"):
            st.markdown(f"**Worked example:** {cat.example}")
            if examples.empty:
                st.caption("No written explanations from students who chose this option.")
            for x in examples.head(3).itertuples():
                ui.html_block(f"<div class='rt-quote'>“{ui.esc(x.explanation)}”</div>"
                              f"<div class='rt-muted'>{ui.esc(x.student_id)} · "
                              f"{ui.esc(x.question_id)} · chose {ui.esc(x.option_text)}</div>")
            st.markdown(f"**Full activity notes:** {cat.review_activity}")
