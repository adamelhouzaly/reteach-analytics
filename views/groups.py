"""Student groups: anonymous small groups built from recurring patterns."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reteach import grouping as gr
from reteach import taxonomy as tx
from ui import components as ui
from ui import state
from ui import theme as th

ui.page_header("Student groups",
               "Students who repeat the same misconception, grouped for small-group "
               "reteaching. Built from the answer key, not machine learning.")
ms = ui.min_support_control("groups")
ctx = state.context()
groups, patterns = ctx.groups, ctx.patterns

if groups.empty:
    ui.empty_state("No students to group yet.", "Upload responses on the Upload data page.")
    st.stop()

pattern_groups = groups[groups["is_pattern"]]
n_grouped = int(pattern_groups["n_students"].sum())

# --- Summary ------------------------------------------------------------------
with ui.card("groups-summary"):
    a, b, c = st.columns(3)
    with a:
        ui.stat(len(pattern_groups), "groups with a recurring pattern")
    with b:
        ui.stat(f"{n_grouped} of {ctx.cov.n_students}", "students in a group")
    with c:
        ui.stat(th.plural(ms, "question"), "needed for a pattern to count")
    colors = [th.TEAL if p else th.SLATE for p in groups["is_pattern"]]
    fig = go.Figure(go.Bar(
        x=groups["n_students"], y=[th.wrap_label(g, 22) for g in groups["group"]],
        orientation="h", marker=dict(color=colors, cornerradius=4),
        text=groups["n_students"], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} students<extra></extra>"))
    rows_h = sum(26 + 16 * g.count("<br>") for g in [th.wrap_label(x, 22)
                                                       for x in groups["group"]])
    th.style_bar_chart(fig, 50 + rows_h + 14 * len(groups), x_title="students",
                       x_max=groups["n_students"].max() * 1.15)
    st.plotly_chart(fig, config=th.PLOTLY_CONFIG, width="stretch")
    st.caption("Gray bars are students without a recurring pattern (not a teaching group).")


def member_table(members: pd.DataFrame) -> pd.DataFrame:
    acc = members["correct"] / members["answered"].where(members["answered"] > 0)
    return pd.DataFrame({
        "Student": members["student_id"],
        "Accuracy": acc.map(th.pct_int),
        "Evidence": members["evidence"].replace("", "—"),
        "Below threshold": members["possible"].map(
            lambda labels: ", ".join(tx.short_name(x) for x in labels) or "—"),
    })


def show_members(members: pd.DataFrame, label: str) -> None:
    with st.expander(label):
        st.dataframe(member_table(members), hide_index=True, width="stretch",
                     height=min(36 * len(members) + 40, 340),
                     column_config={"Accuracy": st.column_config.NumberColumn(format="%d%%"),
                                    "Evidence": st.column_config.TextColumn(width="large")})


st.space("small")
st.markdown("### Teaching groups")
if pattern_groups.empty:
    ui.empty_state("No student repeats a misconception often enough to form a group.",
                   "Try a lower threshold above, or review the Misconceptions page.")

for g in pattern_groups.itertuples():
    members = patterns[patterns["group"] == g.group]
    with ui.card(f"group-{g.Index}"):
        head, count = st.columns([4, 1], vertical_alignment="center")
        with head:
            st.markdown("**Focus:** " + " + ".join(tx.title(m) for m in g.misconceptions))
        with count:
            st.markdown(ui.tag("neutral", th.plural(g.n_students, "student")),
                        unsafe_allow_html=True)
        for m in g.misconceptions:
            st.markdown(f"**Try this:** {tx.CATEGORIES[m].activity}")
        show_members(members, f"Show {th.plural(g.n_students, 'student')}")

st.space("small")
st.markdown("### Not in a teaching group")
for i, (name, note) in enumerate([
    (gr.NO_PATTERN, "Made errors, but no single misconception repeated on enough questions. "
                    "This is not the same as having no difficulty — check their answers."),
    (gr.ALL_CORRECT, "Answered every question they attempted correctly."),
]):
    members = patterns[patterns["group"] == name]
    if members.empty:
        continue
    with ui.card(f"other-{i}"):
        head, count = st.columns([4, 1], vertical_alignment="center")
        with head:
            st.markdown(f"**{name}**")
        with count:
            st.markdown(ui.tag("neutral", th.plural(len(members), "student")),
                        unsafe_allow_html=True)
        ui.muted(note)
        show_members(members, f"Show {th.plural(len(members), 'student')}")

st.download_button(
    "Download all student patterns (CSV)",
    patterns.assign(recurring=patterns["recurring"].str.join("; "),
                    possible=patterns["possible"].str.join("; ")).to_csv(index=False),
    file_name="student_patterns.csv", mime="text/csv", icon=":material/download:")
