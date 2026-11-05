"""Reteach — Student Misconception Analytics (Streamlit presentation layer).

Run:  streamlit run app.py

This file only handles *presentation*. All logic lives in the ``reteach``
package: validation, analytics, grouping, model, report, corrections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reteach import analytics as an
from reteach import corrections as corr
from reteach import grouping as gr
from reteach import model as mdl
from reteach import report
from reteach import taxonomy as tx
from reteach import validation as val

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

# Reference palette (validated light-mode steps).
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
REVIEW_ALL = "Route all to review"

st.set_page_config(page_title="Reteach — Misconception Analytics", page_icon="📐",
                   layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
              font-weight:600; margin-right:6px; border:1px solid transparent;}
      .badge-rule {background:#eef4fc; color:#184f95; border-color:#cde2fb;}
      .badge-ml {background:#fdf0ea; color:#8a3511; border-color:#f6cdb9;}
      .badge-syn {background:#f2f1ee; color:#3d3c39; border-color:#dddcd6;}
      .card {border:1px solid #e4e3df; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem;
             background:#ffffff;}
      .muted {color:#52514e; font-size:0.9rem;}
      div[data-testid="stMetricValue"] {font-size:1.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

RULE = '<span class="badge badge-rule">Teacher-defined rule</span>'
ML = '<span class="badge badge-ml">ML suggestion</span>'
SYN = '<span class="badge badge-syn">Synthetic data</span>'


# --- State -------------------------------------------------------------------
def load_demo() -> dict:
    q = val.validate_questions(DATA / "demo" / "questions.csv")
    r = val.validate_responses(DATA / "demo" / "responses.csv", q.df)
    return {"questions": q.df, "responses": r.df, "label": "Synthetic demo class (60 students)",
            "synthetic": True, "warnings": q.warnings + r.warnings}


def reset_state() -> None:
    st.session_state.dataset = load_demo()
    st.session_state.corrections = []
    st.session_state.pop("review_editor", None)


if "dataset" not in st.session_state:
    reset_state()


@st.cache_resource(show_spinner="Training the explanation classifier from bundled data…")
def get_model() -> mdl.TrainedModel:
    """Load the locally trained model, or train it from the bundled labeled data.
    Uploaded files are never deserialised."""
    trained = mdl.load_local()
    if trained is None:
        from reteach.evaluate import train_model
        trained, _ = train_model()
    return trained


@st.cache_data
def load_metrics() -> dict | None:
    path = ARTIFACTS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
        hoverlabel=dict(bgcolor="white", font_size=13), plot_bgcolor="#fcfcfb",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#ecebe7", zerolinecolor="#dddcd6")
    fig.update_yaxes(gridcolor="#ecebe7", zerolinecolor="#dddcd6")
    return fig


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.0%}"


# --- Sidebar -----------------------------------------------------------------
ds = st.session_state.dataset
with st.sidebar:
    st.markdown("### 📐 Reteach")
    st.caption("Student misconception analytics for fraction addition.")
    st.markdown(f"**Dataset:** {ds['label']}")
    if ds["synthetic"]:
        st.markdown(SYN, unsafe_allow_html=True)
        st.caption("All students, answers and explanations are fictional.")
    min_support = st.slider(
        "Minimum supporting questions", 1, 5, an.DEFAULT_MIN_SUPPORT,
        help="A misconception counts as *recurring* for a student only if they chose a matching "
             "distractor on at least this many distinct questions. One wrong answer is evidence, "
             "not a diagnosis.")
    st.divider()
    if st.button("Reset demo", use_container_width=True,
                 help="Reload the synthetic demo class and clear unsaved corrections."):
        reset_state()
        st.rerun()
    st.caption("Legend:")
    st.markdown(RULE + "<br><span class='muted'>from the answer key</span>", unsafe_allow_html=True)
    st.markdown(ML + "<br><span class='muted'>from the text classifier</span>",
                unsafe_allow_html=True)

questions, responses = ds["questions"], ds["responses"]
scored = an.score_responses(questions, responses)
cov = an.coverage(questions, scored)

st.title("Reteach — Student Misconception Analytics")
st.markdown(
    "Find fraction concepts worth reteaching, see the evidence behind each pattern, and review "
    "machine-suggested categories for written explanations."
    + (" &nbsp;" + SYN if ds["synthetic"] else ""), unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Misconceptions", "Student groups", "Written-answer review",
                "Model evaluation", "Data upload"])


def safe(render):
    """Render a tab, showing a friendly message instead of a traceback."""
    try:
        render()
    except Exception as exc:  # noqa: BLE001 - deliberate UI guard
        st.error("Something went wrong while rendering this view. Try *Reset demo* or check the "
                 f"uploaded files on the Data upload tab. ({type(exc).__name__})")


# --- Classification of written explanations ---------------------------------
def explanation_rows() -> pd.DataFrame:
    """Responses to questions where at least one student wrote an explanation."""
    asked = set(scored.loc[scored["explanation"] != "", "question_id"])
    rows = scored[scored["question_id"].isin(asked)].copy()
    return rows.reset_index(drop=True)


def classified_rows(threshold: float) -> pd.DataFrame:
    rows = explanation_rows()
    if rows.empty:
        return rows
    model = get_model()
    model_at_t = mdl.TrainedModel(model.pipeline, threshold, model.meta)
    c = mdl.classify(model_at_t, rows["explanation"])
    out = pd.concat([rows[["student_id", "question_id", "option_text", "outcome",
                           "misconception_label"]], c], axis=1)
    return out


# --- Tabs ------------------------------------------------------------------
def tab_overview():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", cov.n_students)
    c2.metric("Response coverage", pct(cov.response_rate),
              help=f"{cov.answered} answered of {cov.expected} expected "
                   f"({cov.n_students} students × {cov.n_questions} questions).")
    c3.metric("Accuracy (answered)", pct(cov.accuracy),
              help=f"{cov.correct} correct of {cov.answered} answered. Blank and missing responses "
                   "are excluded from accuracy.")
    c4.metric("Blank / missing", f"{cov.unanswered} / {cov.missing}",
              help="Blank = row present but no option chosen. Missing = no row for that student "
                   "and question. Neither counts as wrong.")

    topics = an.topic_summary(scored)
    mis = an.misconception_summary(questions, scored, min_support)
    suggestions = an.suggest_topics(topics, mis, cov.n_students, questions)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Accuracy by topic")
        fig = px.bar(topics, x="accuracy", y="topic", orientation="h",
                     text=topics.apply(lambda r: f"{r.accuracy:.0%}  ({r.correct}/{r.answered})",
                                       axis=1),
                     custom_data=["correct", "answered", "questions"])
        fig.update_traces(marker_color=BLUE, textposition="inside", insidetextanchor="end",
                          textfont_color="white",
                          hovertemplate="%{y}<br>%{x:.0%} correct (%{customdata[0]} of "
                                        "%{customdata[1]} answered)<br>%{customdata[2]} questions"
                                        "<extra></extra>")
        fig.add_vline(x=an.TOPIC_ACCURACY_THRESHOLD, line_dash="dot", line_color=MUTED,
                      annotation_text=f"review below {an.TOPIC_ACCURACY_THRESHOLD:.0%}",
                      annotation_font_color=MUTED)
        fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title=None)
        fig.update_yaxes(title=None, autorange="reversed")
        st.plotly_chart(fig_style(fig, 260), use_container_width=True)

        st.subheader("Most common error patterns")
        st.markdown(RULE, unsafe_allow_html=True)
        common = mis.sort_values("selections", ascending=False)
        for r in common.itertuples():
            st.markdown(f"**{r.name}** — chosen {r.selections} times out of {r.opportunities} "
                        f"opportunities ({pct(r.selection_rate)}); recurring for "
                        f"{r.students_recurring} of {cov.n_students} students.")
        un = an.unmapped_error_summary(scored)
        st.caption(f"Unmapped wrong answers (no misconception assigned): {un['count']} of "
                   f"{un['answered']} answered responses.")

    with right:
        st.subheader("Suggested topics to review")
        st.markdown(RULE, unsafe_allow_html=True)
        st.caption(f"Suggested when accuracy < {an.TOPIC_ACCURACY_THRESHOLD:.0%} or a linked "
                   f"misconception recurs for ≥ {an.MISCONCEPTION_SHARE_THRESHOLD:.0%} of students. "
                   "Activities are authored guidance, not AI-generated.")
        if not suggestions:
            st.success("No topic met the review rules.")
        for s in suggestions:
            with st.container(border=True):
                st.markdown(f"**{s['topic']}** · accuracy {pct(s['accuracy'])}")
                for reason in s["reasons"]:
                    st.markdown(f"- {reason}")
                for act in s["activities"]:
                    st.markdown(f"> {act}")

    st.subheader("Accuracy by question")
    qs = an.question_summary(questions, scored)
    st.dataframe(
        qs[["question_id", "topic", "prompt", "correct_answer", "correct", "answered", "accuracy",
            "unanswered", "missing", "most_common_wrong"]],
        hide_index=True, use_container_width=True,
        column_config={"accuracy": st.column_config.ProgressColumn(
            "accuracy", format="percent", min_value=0, max_value=1)})

    st.subheader("Export")
    threshold = st.session_state.get("threshold_value", get_model().threshold)
    md = report.build_report(questions, scored, min_support, classified_rows(threshold),
                             ds["label"], ds["synthetic"])
    st.download_button("Download reteaching report (Markdown)", md,
                       file_name="reteach_report.md", mime="text/markdown", type="primary")
    with st.expander("Preview report"):
        st.markdown(md)


def tab_misconceptions():
    st.markdown(RULE + " Counts come from the misconception labels the teacher assigned to "
                "each distractor in the answer key. No machine learning is used here.",
                unsafe_allow_html=True)
    mis = an.misconception_summary(questions, scored, min_support)
    patterns = gr.student_evidence(questions, scored)
    fig = go.Figure(go.Bar(
        x=mis["selection_rate"], y=mis["name"], orientation="h", marker_color=BLUE,
        text=[f"{s} / {o}" for s, o in zip(mis["selections"], mis["opportunities"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>chosen on %{x:.0%} of opportunities<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.1, mis["selection_rate"].fillna(0).max())
                                              * 1.3], title="share of opportunities (chosen / "
                                                            "answered questions offering it)")
    fig.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(fig_style(fig, 260), use_container_width=True)

    for r in mis.itertuples():
        cat = tx.CATEGORIES[r.misconception]
        with st.container(border=True):
            st.markdown(f"#### {cat.name}")
            st.markdown(f"<span class='muted'>{cat.description}</span>", unsafe_allow_html=True)
            st.markdown(f"Worked example: `{cat.example}`")
            a, b, c = st.columns(3)
            a.metric("Chosen / opportunities", f"{r.selections} / {r.opportunities}",
                     help="Opportunities = answered responses to questions that offer this "
                          "distractor.")
            b.metric("Students (at least once)", f"{r.students_any} / {r.students_with_opportunity}",
                     help="Denominator = students who answered at least one supporting question.")
            c.metric(f"Students (≥ {min_support} questions)", r.students_recurring)
            sup = questions[(questions["misconception_label"] == r.misconception)]
            st.markdown("**Supporting questions:** " + "; ".join(
                f"{x.question_id} → option {x.option_id} “{x.option_text}”"
                for x in sup.itertuples()))
            ev = patterns[(patterns["misconception"] == r.misconception)
                          & (patterns["n_supporting"] > 0)]
            ev = ev.sort_values("n_supporting", ascending=False)
            rec = ev[ev["n_supporting"] >= min_support]
            pos = ev[ev["n_supporting"] < min_support]
            st.markdown("**Recurring:** " + (", ".join(
                f"{x.student_id} ({x.n_supporting}/{x.opportunities})" for x in rec.itertuples())
                or "none"))
            st.markdown(f"**Possible — insufficient evidence (< {min_support} questions):** "
                        + (", ".join(f"{x.student_id} ({x.n_supporting}/{x.opportunities})"
                                     for x in pos.itertuples()) or "none"))
            ex = scored[(scored["misconception_label"] == r.misconception)
                        & (scored["explanation"] != "")]
            if len(ex):
                st.markdown("**Example explanations from students who chose it:**")
                for x in ex.head(3).itertuples():
                    st.markdown(f"- {x.student_id}, {x.question_id} (chose {x.option_text}): "
                                f"“{x.explanation}”")
            st.markdown(f"**Suggested activity** (authored guidance): {cat.review_activity}")


def tab_groups():
    st.markdown(RULE + f" Students are grouped by the misconceptions that recur on at least "
                f"**{min_support}** distinct questions (change this in the sidebar). IDs are "
                "anonymous.", unsafe_allow_html=True)
    patterns = gr.student_patterns(questions, scored, min_support)
    groups = gr.groups(patterns)
    fig = px.bar(groups, x="n_students", y="group", orientation="h", text="n_students")
    fig.update_traces(marker_color=[BLUE if p else "#b9b8b2" for p in groups["is_pattern"]],
                      textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x} students<extra></extra>")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="students", range=[0, groups["n_students"].max() * 1.15])
    st.plotly_chart(fig_style(fig, 60 + 36 * len(groups)), use_container_width=True)

    for g in groups.itertuples():
        members = patterns[patterns["group"] == g.group]
        with st.expander(f"{g.group} — {g.n_students} student(s)", expanded=g.is_pattern):
            if g.is_pattern:
                acts = [tx.CATEGORIES[m].review_activity for m in g.misconceptions]
                st.markdown("**Suggested small-group focus** (authored guidance): "
                            + " ".join(acts))
            else:
                st.caption("No misconception met the minimum-support rule for these students. "
                           "Some may still show a possible pattern on a single question.")
            st.dataframe(
                members.assign(possible=members["possible"].map(
                    lambda labels: ", ".join(tx.short_name(x) for x in labels)))
                [["student_id", "evidence", "possible", "answered", "correct",
                  "unmapped_errors"]].rename(columns={
                    "evidence": "evidence (questions chosen / opportunities)",
                    "possible": "below threshold"}),
                hide_index=True, use_container_width=True)
    st.download_button("Download student patterns (CSV)",
                       patterns.assign(recurring=patterns["recurring"].str.join("; "),
                                       possible=patterns["possible"].str.join("; "))
                       .to_csv(index=False), file_name="student_patterns.csv", mime="text/csv")


def threshold_control(model: mdl.TrainedModel, metrics: dict | None) -> float:
    grid = [f"{t:.2f}" for t in mdl.THRESHOLD_GRID]
    default = REVIEW_ALL if model.threshold > 1 else f"{model.threshold:.2f}"
    choice = st.select_slider("Auto-accept threshold (model score)", options=grid + [REVIEW_ALL],
                              value=default if default in grid + [REVIEW_ALL] else REVIEW_ALL,
                              help="Predictions scoring below this go to teacher review.")
    t = 1.01 if choice == REVIEW_ALL else float(choice)
    st.session_state.threshold_value = t
    if metrics:
        table = pd.DataFrame(metrics["training"]["threshold_table_validation"])
        row = table[table["threshold"].round(2) == round(t, 2)]
        if len(row):
            r = row.iloc[0]
            st.caption(f"On validation data this threshold auto-accepts {r.coverage:.0%} of "
                       f"predictions with {pct(r.accepted_accuracy)} accuracy on those accepted.")
        st.caption(f"Default chosen from validation data: "
                   f"{'route all to review' if model.threshold > 1 else f'{model.threshold:.2f}'} "
                   f"(policy: {metrics['training']['threshold_policy']}). Scores are uncalibrated.")
    return t


def tab_review():
    st.markdown(ML + " A TF-IDF + logistic-regression model suggests a category for each written "
                "explanation. Suggestions are not diagnoses; corrections are stored separately and "
                "never retrain the model automatically.", unsafe_allow_html=True)
    model, metrics = get_model(), load_metrics()
    left, right = st.columns([1, 1])
    with left:
        t = threshold_control(model, metrics)
    rows = classified_rows(t)
    if rows.empty:
        st.info("This dataset has no written explanations.")
        return
    with right:
        counts = rows["state"].value_counts()
        st.markdown("**Handling states**")
        for state in [tx.STATE_AUTO, tx.STATE_NEEDS_REVIEW, tx.STATE_UNRECOGNIZED,
                      tx.STATE_INSUFFICIENT, tx.STATE_BLANK]:
            st.markdown(f"- {tx.display_name(state)}: **{int(counts.get(state, 0))}** — "
                        f"<span class='muted'>{tx.STATE_DESCRIPTIONS[state]}</span>",
                        unsafe_allow_html=True)

    states = st.multiselect(
        "Show", [tx.STATE_NEEDS_REVIEW, tx.STATE_AUTO, tx.STATE_UNRECOGNIZED, tx.STATE_INSUFFICIENT,
                 tx.STATE_BLANK], default=[tx.STATE_NEEDS_REVIEW, tx.STATE_AUTO,
                                           tx.STATE_UNRECOGNIZED],
        format_func=tx.display_name)
    view = rows[rows["state"].isin(states)].sort_values("score", na_position="last")
    view = view.reset_index(drop=True)
    corrected = {(c["student_id"], c["question_id"]): c["corrected_label"]
                 for c in st.session_state.corrections}
    table = pd.DataFrame({
        "student_id": view["student_id"],
        "question_id": view["question_id"],
        "answer": view["option_text"],
        "explanation": view["explanation"],
        "suggested category (ML)": view["predicted_label"].map(
            lambda x: tx.short_name(x) if x else "—"),
        "model score": view["score"],
        "review status": view["state"].map(tx.display_name),
        "teacher label": [corrected.get(k, "") for k in zip(view["student_id"],
                                                             view["question_id"])],
    })
    table["teacher label"] = table["teacher label"].map(lambda x: tx.short_name(x) if x else "")
    st.dataframe(
        table, hide_index=True, use_container_width=True, height=360,
        column_config={
            "model score": st.column_config.NumberColumn(format="%.2f",
                                                         help="Uncalibrated max probability."),
            "explanation": st.column_config.TextColumn(width="large"),
        })
    st.caption("Sorted by model score, lowest (least certain) first.")

    if view.empty:
        return
    with st.form("correction_form", border=True):
        st.markdown("**Correct a label**")
        keys = [f"{r.student_id} · {r.question_id} · “{r.explanation[:70]}”"
                for r in view.itertuples()]
        pick = st.selectbox("Explanation", range(len(view)), format_func=lambda i: keys[i])
        row = view.iloc[pick]
        label_options = [*tx.TEXT_LABELS, tx.STATE_INSUFFICIENT]
        default = (label_options.index(row["predicted_label"])
                   if row["predicted_label"] in label_options else 0)
        new_label = st.selectbox("Correct category", label_options, index=default,
                                 format_func=tx.display_name)
        submitted = st.form_submit_button("Save correction", type="primary")
    if submitted:
        row = view.iloc[pick]
        new = [corr.make_correction(row.to_dict(), new_label, dataset=ds["label"])]
        st.session_state.corrections.extend(new)
        path = corr.append_to_file(new)
        st.success(f"Saved correction for {row['student_id']} · {row['question_id']} → "
                   f"{tx.display_name(new_label)} in {path.name} "
                   "(source = teacher_correction, used_for_training = False).")
    c1, c2 = st.columns([1, 2])
    if st.session_state.corrections:
        c2.download_button("Download corrections (CSV)",
                           corr.to_frame(st.session_state.corrections).to_csv(index=False),
                           file_name="teacher_corrections.csv", mime="text/csv")
        st.caption(f"{len(st.session_state.corrections)} correction(s) this session. A future "
                   "retraining run would require a second review, new group IDs, and a full "
                   "re-evaluation before replacing the model.")


def tab_evaluation():
    metrics = load_metrics()
    if metrics is None:
        st.warning("No evaluation results found. Run `python -m reteach.cli evaluate`.")
        return
    st.warning("**Synthetic evaluation.** " + metrics["data_note"] + " The labeled explanations "
               "were written by the project author from templates; there is no expert annotation.")
    t, d = metrics["test"], metrics["dataset"]
    a, b, c, e = st.columns(4)
    a.metric("Test accuracy", pct(t["model"]["accuracy"]),
             help=f"Most-frequent baseline: {pct(t['baseline_most_frequent']['accuracy'])}")
    b.metric("Test macro F1", f"{t['model']['macro_f1']:.2f}",
             help=f"Most-frequent baseline: {t['baseline_most_frequent']['macro_f1']:.2f}")
    c.metric("Test examples / families", f"{d['test']['examples']} / {d['test']['families']}")
    s = t["selective_prediction"]
    e.metric("Auto-accepted at default threshold", pct(s["coverage"]),
             help="Share of test predictions that would skip teacher review.")
    st.caption(f"Baseline (always predicts the most frequent class): accuracy "
               f"{pct(t['baseline_most_frequent']['accuracy'])}, macro F1 "
               f"{t['baseline_most_frequent']['macro_f1']:.2f}.")

    left, right = st.columns(2)
    with left:
        st.subheader("Model vs. most-frequent baseline")
        comp = pd.DataFrame({
            "metric": ["Accuracy", "Macro F1"] * 2,
            "value": [t["model"]["accuracy"], t["model"]["macro_f1"],
                      t["baseline_most_frequent"]["accuracy"],
                      t["baseline_most_frequent"]["macro_f1"]],
            "system": ["TF-IDF + LogReg"] * 2 + ["Most-frequent baseline"] * 2})
        fig = px.bar(comp, x="metric", y="value", color="system", barmode="group",
                     text=comp["value"].map(lambda v: f"{v:.2f}"),
                     color_discrete_sequence=[BLUE, ORANGE])
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f}<extra></extra>")
        fig.update_yaxes(range=[0, 1.1], title=None)
        fig.update_xaxes(title=None)
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None), bargap=0.35,
                          bargroupgap=0.08)
        st.plotly_chart(fig_style(fig, 320), use_container_width=True)
    with right:
        st.subheader("Confusion matrix (test)")
        labels = t["confusion_matrix"]["labels"]
        names = [tx.short_name(x) for x in labels]
        fig = go.Figure(go.Heatmap(
            z=t["confusion_matrix"]["matrix"], x=names, y=names, colorscale=SEQ_BLUE,
            text=t["confusion_matrix"]["matrix"], texttemplate="%{text}", showscale=False,
            xgap=2, ygap=2,
            hovertemplate="true %{y}<br>predicted %{x}<br>%{z} examples<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="true")
        fig.update_xaxes(title="predicted", side="bottom")
        st.plotly_chart(fig_style(fig, 340), use_container_width=True)

    st.subheader("Per-class results (test)")
    pc = pd.DataFrame(t["model"]["per_class"]).T.reset_index(names="class")
    pc["test families"] = pc["class"].map(lambda x: d["test"]["by_label"][x]["families"])
    pc["class"] = pc["class"].map(tx.display_name)
    st.dataframe(pc, hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="%.2f")
                                for k in ["precision", "recall", "f1"]})

    st.subheader("Review threshold")
    st.markdown(
        f"The threshold is chosen on **out-of-fold validation predictions** from the development "
        f"split ({metrics['training']['threshold_policy']}). "
        + ("No validation threshold met that policy, so **every prediction is routed to teacher "
           "review** by default." if s["threshold"] > 1 else
           f"Chosen: {s['threshold']:.2f}.")
        + f" On test: coverage {pct(s['coverage'])}, accuracy on accepted "
          f"{pct(s['accepted_accuracy'])}, accuracy on routed {pct(s['review_accuracy'])}.")
    table = pd.DataFrame(metrics["training"]["threshold_table_validation"])
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"coverage": st.column_config.NumberColumn(format="percent"),
                                "accepted_accuracy": st.column_config.NumberColumn(
                                    format="percent")})

    st.subheader("Representative errors (test)")
    preds_path = ARTIFACTS / "test_predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        err = preds[preds["label"] != preds["predicted"]]
        st.dataframe(err[["explanation", "label", "predicted", "score"]].head(15),
                     hide_index=True, use_container_width=True)
    st.subheader("Dataset and limitations")
    st.markdown(
        f"- Development: {d['dev']['examples']} examples from {d['dev']['families']} template "
        f"families; test: {d['test']['examples']} examples from {d['test']['families']} unseen "
        f"families on held-out question contexts ({', '.join(d['test']['contexts'])}).\n"
        "- Paraphrases of one family always share a split (group-aware), so the test set measures "
        "generalisation to *new reasoning wording*, not memorised sentences.\n"
        "- The text is author-written and cleaner than real student writing; real performance "
        "would likely differ and must be measured on real, consented, expert-labelled data.\n"
        "- Scores are uncalibrated. Calibration was not evaluated.\n"
        "- Full write-up: `docs/evaluation.md` and `docs/model_card.md`.")


def tab_upload():
    st.markdown("Upload your own quiz. Files are validated before use and are never executed or "
                "deserialised — only CSV text is read.")
    with st.container(border=True):
        st.markdown("**1. Download templates**")
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Questions template", (DATA / "templates" /
                                                  "questions_template.csv").read_bytes(),
                           "questions_template.csv", "text/csv")
        c2.download_button("Responses template", (DATA / "templates" /
                                                  "responses_template.csv").read_bytes(),
                           "responses_template.csv", "text/csv")
        c3.download_button("Demo questions", (DATA / "demo" / "questions.csv").read_bytes(),
                           "demo_questions.csv", "text/csv")
        c4.download_button("Messy sample responses", (DATA / "invalid" /
                                                      "responses_messy.csv").read_bytes(),
                           "responses_messy.csv", "text/csv")
    with st.expander("Schema guidance", expanded=False):
        for name, cols in val.SCHEMA_DOCS.items():
            st.markdown(f"**{name}.csv**")
            st.dataframe(pd.DataFrame(cols.items(), columns=["column", "meaning"]),
                         hide_index=True, use_container_width=True)
        st.markdown(
            "**How missing data is treated:** a blank `selected_option` is *unanswered*; a "
            "student/question pair with no row is *missing*. Neither counts toward accuracy; both "
            "are reported as coverage. Duplicate student/question rows keep the last row.")

    st.markdown("**2. Upload files**")
    qfile = st.file_uploader("Questions CSV (one row per option)", type=["csv"], key="q_upload")
    rfile = st.file_uploader("Responses CSV", type=["csv"], key="r_upload")
    use_demo_q = st.checkbox("Use the demo questions instead of uploading", value=qfile is None)
    if st.button("Validate", type="primary", disabled=rfile is None):
        qres = (val.validate_questions(DATA / "demo" / "questions.csv") if use_demo_q or
                qfile is None else val.validate_questions(qfile.getvalue()))
        show_messages("Questions", qres)
        if not qres.ok:
            return
        rres = val.validate_responses(rfile.getvalue(), qres.df)
        show_messages("Responses", rres)
        if rres.ok:
            st.session_state.pending = {
                "questions": qres.df, "responses": rres.df,
                "label": f"Uploaded: {rfile.name}", "synthetic": False,
                "warnings": qres.warnings + rres.warnings}
            st.success(f"Ready: {rres.df['student_id'].nunique()} students, "
                       f"{len(rres.df)} response rows.")
    if "pending" in st.session_state and st.button("Use uploaded data in the dashboard"):
        st.session_state.dataset = st.session_state.pop("pending")
        st.session_state.corrections = []
        st.rerun()


def show_messages(name: str, res: val.ValidationResult) -> None:
    if res.errors:
        st.error(f"**{name}: {len(res.errors)} problem(s) must be fixed.**\n\n"
                 + "\n".join(f"- {e}" for e in res.errors))
    if res.warnings:
        st.warning(f"**{name}: {len(res.warnings)} warning(s).**\n\n"
                   + "\n".join(f"- {w}" for w in res.warnings))
    if res.ok and not res.warnings:
        st.success(f"{name}: no problems found.")


for tab, render in zip(tabs, [tab_overview, tab_misconceptions, tab_groups, tab_review,
                              tab_evaluation, tab_upload]):
    with tab:
        safe(render)
