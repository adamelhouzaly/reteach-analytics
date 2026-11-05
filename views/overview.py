"""Overview: class performance and the next teaching action at a glance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reteach import analytics as an
from reteach import report
from reteach import taxonomy as tx
from ui import components as ui
from ui import state
from ui import theme as th

ctx = state.context()
cov = ctx.cov
rows = state.classified(ctx)
n_review = int(rows["to_review"].sum()) if len(rows) else 0

ui.page_header("Class overview")

# --- Summary cards (2-up on phones, 4-up on wider screens) ------------------------------
with st.container(horizontal=True, gap="small", wrap=True):
    with ui.card("kpi-students"):
        st.metric("Students", cov.n_students,
                  help=f"Students with at least one answer. Quiz: {cov.n_questions} questions.")
    with ui.card("kpi-accuracy"):
        st.metric("Class accuracy", th.pct(cov.accuracy),
                  help=f"{cov.correct} correct of {cov.answered} answered. Blank and missing "
                       "answers are not counted as wrong.")
    with ui.card("kpi-coverage"):
        st.metric("Response coverage", th.pct(cov.response_rate),
                  help=f"{cov.answered} answered of {cov.expected} expected "
                       f"({cov.n_students} students × {cov.n_questions} questions). "
                       f"{cov.unanswered} left blank, {cov.missing} missing.")
    with ui.card("kpi-review"):
        st.metric("Answers to review", n_review,
                  help="Written explanations whose model suggestion still needs a teacher's "
                       "check.")
        if n_review and st.button("Open queue", key="open_review", type="tertiary",
                                  icon=":material/arrow_forward:"):
            ui.go("review")

st.space("small")

# --- What to reteach next -------------------------------------------------------------
mis_by_label = ctx.misconceptions.set_index("misconception")
if ctx.suggestions:
    top = ctx.suggestions[0]
    linked = top["linked_misconceptions"]
    topic_row = ctx.topics.set_index("topic").loc[top["topic"]]
    with ui.card("next", variant="priority"):
        ui.eyebrow("What to reteach next")
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            ui.html_block(f"<div class='rt-big'>{ui.esc(top['topic'])}</div>")
            ui.muted(f"{th.pct(top['accuracy'])} correct · {int(topic_row['correct'])} of "
                     f"{int(topic_row['answered'])} answers · "
                     f"{th.plural(int(topic_row['questions']), 'question')}")
            st.space("small")
            st.markdown("**Why this topic**")
            evidence = []
            if top["accuracy"] < an.TOPIC_ACCURACY_THRESHOLD:
                evidence.append(f"Below the {an.TOPIC_ACCURACY_THRESHOLD:.0%} accuracy target")
            for label in linked:
                k = int(mis_by_label.loc[label, "students_recurring"])
                evidence.append(f"{tx.title(label)}: recurring for {k} of {cov.n_students} "
                                "students")
            st.markdown("\n".join(f"- {e}" for e in evidence))
        with right:
            st.markdown("**Try this** &nbsp;" + ui.tag("rule", "Authored guidance"),
                        unsafe_allow_html=True)
            if linked:
                for label in linked:
                    st.markdown(f"**{tx.title(label)}.** {tx.CATEGORIES[label].activity}")
            else:
                st.markdown("No single misconception explains these errors. Open "
                            "*Results by question* below to see which items were missed.")
            if linked:
                with st.container(horizontal=True, gap="small", wrap=True):
                    if st.button("See misconception evidence", type="primary", key="goto_mis"):
                        st.session_state.focus_misconception = linked[0]
                        ui.go("misconceptions")
                    if st.button("See student groups", key="goto_groups"):
                        ui.go("groups")
    others = ctx.suggestions[1:]
    if others:
        st.caption("Also below target: " + "; ".join(
            f"{s['topic']} ({th.pct(s['accuracy'])})" for s in others))
else:
    ui.empty_state("No topic needs reteaching under the current rules.",
                   f"Topics are suggested when accuracy is below "
                   f"{an.TOPIC_ACCURACY_THRESHOLD:.0%} or a misconception recurs for at least "
                   f"{an.MISCONCEPTION_SHARE_THRESHOLD:.0%} of students.")

st.space("small")

# --- Topic chart + common mistakes ---------------------------------------------
left, right = st.columns([1.1, 1], gap="medium")
with left, ui.card("topics"):
    st.markdown("### Accuracy by topic")
    t = ctx.topics
    priority = ctx.suggestions[0]["topic"] if ctx.suggestions else None
    fig = go.Figure(go.Bar(
        x=t["accuracy"],
        y=[f"{th.wrap_label(x, 22)}<br><b>{a:.0%}</b>" for x, a in zip(t["topic"],
                                                                  t["accuracy"])],
        orientation="h",
        marker=dict(color=[th.AMBER_BAR if x == priority else th.TEAL for x in t["topic"]],
                    cornerradius=4),
        customdata=t[["correct", "answered"]],
        hovertemplate="%{x:.0%} correct (%{customdata[0]} of %{customdata[1]} answers)"
                      "<extra></extra>"))
    fig.add_vline(x=an.TOPIC_ACCURACY_THRESHOLD, line=dict(color=th.MUTED, width=1, dash="dot"))
    th.style_bar_chart(fig, 60 + 66 * len(t), percent=True, x_max=1.0)
    fig.update_xaxes(tickvals=[0, 0.25, 0.5, 0.7, 1.0])
    st.plotly_chart(fig, config=th.PLOTLY_CONFIG, width="stretch")
    st.caption(f"Amber = suggested next topic. Dotted line = "
               f"{an.TOPIC_ACCURACY_THRESHOLD:.0%} target. Blank and missing answers excluded.")

with right, ui.card("mistakes"):
    st.markdown("### Common mistakes &nbsp;" + ui.tag("rule"), unsafe_allow_html=True)
    mis = ctx.misconceptions.sort_values(["students_recurring", "selections"],
                                         ascending=False)
    items = "".join(
        f"<div class='rt-item'><div class='rt-item-title'>{ui.esc(tx.title(r.misconception))}"
        f"</div><div class='rt-muted'>Recurring for {r.students_recurring} of {cov.n_students} "
        f"students · chosen {r.selections} of {r.opportunities} times "
        f"({th.pct(r.selection_rate)})</div></div>"
        for r in mis.itertuples())
    ui.html_block(items)
    un = an.unmapped_error_summary(ctx.scored)
    st.caption(f"Recurring = same mistake on {th.plural(ctx.min_support, 'question')} or more. "
               f"{un['count']} other wrong answers are not linked to a misconception.")

# --- Details ---------------------------------------------------------------------
st.space("small")
with st.expander("Results by question"):
    qs = ctx.questions_summary
    table = pd.DataFrame({
        "Question": qs["question_id"],
        "Topic": qs["topic"],
        "Accuracy": qs["accuracy"].map(th.pct_int),
        "Correct": qs["correct"],
        "Answered": qs["answered"],
        "Correct answer": qs["correct_answer"],
        "Most common wrong answer": qs["most_common_wrong"],
        "Blank": qs["unanswered"],
        "Missing": qs["missing"],
        "Prompt": qs["prompt"],
    })
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Accuracy": st.column_config.NumberColumn(format="%d%%"),
        "Prompt": st.column_config.TextColumn(width="large"),
    })
    st.caption("Accuracy = correct ÷ answered. Blank = row present but no option chosen; "
               "Missing = no row for that student.")

with st.expander("Reteaching report"):
    ds = state.dataset()
    md = report.build_report(ctx.questions, ctx.scored, ctx.min_support, rows, ds["label"],
                             ds["synthetic"])
    st.download_button("Download report (Markdown)", md, file_name="reteach_report.md",
                       mime="text/markdown", type="primary", icon=":material/download:")
    with st.container(height=420, border=True):
        st.markdown(md)
