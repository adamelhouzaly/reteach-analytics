"""Review answers: confirm or change the model's suggested category, one answer at a time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reteach import corrections as corr
from reteach import model as mdl
from reteach import taxonomy as tx
from ui import components as ui
from ui import state
from ui import theme as th

REVIEW_ALL = "Send all to review"
LABEL_OPTIONS = [*tx.TEXT_LABELS, tx.STATE_INSUFFICIENT]
STATUS_TEXT = {
    tx.STATE_NEEDS_REVIEW: ("attention", "Needs review"),
    tx.STATE_UNRECOGNIZED: ("attention", "Unrecognized wording"),
    tx.STATE_AUTO: ("neutral", "Auto-accepted"),
    tx.STATE_BLANK: ("neutral", "No explanation"),
    tx.STATE_INSUFFICIENT: ("neutral", "Too short to judge"),
}


def label_title(label: str) -> str:
    return "Too little to judge" if label == tx.STATE_INSUFFICIENT else tx.title(label)


ui.page_header("Review answers",
               "Check the category the model suggests for each written explanation. Your "
               "confirmed labels are saved separately and never retrain the model.")
ctx = state.context()
rows = state.classified(ctx)
if rows.empty:
    ui.empty_state("This dataset has no written explanations to review.",
                   "Add an 'explanation' column to your responses file to use this page.")
    st.stop()

confirmed = rows["teacher_label"] != ""
FILTERS = {
    "To review": rows["to_review"],
    "Confirmed": confirmed,
    "Auto-accepted": (rows["state"] == tx.STATE_AUTO) & ~confirmed,
    "Blank or too short": rows["state"].isin([tx.STATE_BLANK, tx.STATE_INSUFFICIENT])
    & ~confirmed,
    "All": pd.Series(True, index=rows.index),
}
st.session_state.setdefault("review_filter", "To review")
choice = st.segmented_control(
    "Show", list(FILTERS), key="review_filter", required=True,
    format_func=lambda f: f"{f} ({int(FILTERS[f].sum())})")

view = rows[FILTERS[choice]]
if choice == "To review":
    # Least certain first; ties broken by question/student so the order never shuffles.
    view = view.sort_values(["score", "question_id", "student_id"], na_position="last",
                            kind="stable")
else:
    view = view.sort_values(["question_id", "student_id"], kind="stable")
view = view.reset_index(drop=True)

if view.empty:
    msg = ("Nothing left to review — every flagged answer has a teacher-confirmed label."
           if choice == "To review" else "No answers in this list.")
    ui.empty_state(msg, "Use the filter above to see other answers.")
else:
    keys = [f"{r.student_id}|{r.question_id}" for r in view.itertuples()]
    current = st.session_state.get("review_key")
    if current not in keys:
        pos = min(st.session_state.get("review_pos", 0), len(keys) - 1)
        st.session_state.review_key = keys[pos]
    i = keys.index(st.session_state.review_key)
    st.session_state.review_pos = i
    row = view.iloc[i]

    def move(step: int) -> None:
        st.session_state.review_key = keys[(i + step) % len(keys)]

    def save(row: dict, filt: str) -> None:
        label = st.session_state.get(f"cat_{keys[i]}")
        if label is None:
            state.flash("Choose a category before saving.", "warning")
            return
        c = corr.make_correction(row, label, dataset=state.dataset()["label"])
        st.session_state.corrections.append(c)
        corr.append_to_file([c])
        state.flash(f"Saved {row['student_id']} · {row['question_id']} as "
                    f"“{label_title(label)}”.")
        if filt == "To review":  # the saved answer leaves this list; show the next one
            nxt = keys[i + 1] if i + 1 < len(keys) else (keys[i - 1] if i > 0 else None)
        else:
            nxt = keys[min(i + 1, len(keys) - 1)]
        if nxt:
            st.session_state.review_key = nxt

    left, right = st.columns([1.35, 1], gap="medium")
    with left, ui.card("review-item"):
        nav = st.container(horizontal=True, vertical_alignment="center", gap="small")
        with nav:
            ui.eyebrow(f"Answer {i + 1} of {len(keys)}")
            st.space("stretch")
            st.button("Previous", key="prev", on_click=move, args=(-1,),
                      icon=":material/chevron_left:", type="tertiary")
            st.button("Next", key="next", on_click=move, args=(1,),
                      icon=":material/chevron_right:", type="tertiary")

        kind, status = STATUS_TEXT.get(row["state"], ("neutral", row["state"]))
        if row["teacher_label"]:
            kind, status = "teacher", "Teacher-confirmed"
        st.markdown(f"### {row['student_id']} · {row['question_id']}  "
                    + ui.tag(kind, status), unsafe_allow_html=True)
        ui.muted(row["prompt"])

        if row["outcome"] == "unanswered":
            key_note = ui.tag("neutral", "No option chosen")
        elif row["is_correct"]:
            key_note = ui.tag("rule", "Correct answer")
        elif row["misconception_label"]:
            key_note = ui.tag("rule", f"Answer key: {tx.title(row['misconception_label'])}")
        else:
            key_note = ui.tag("neutral", "Wrong, not linked to a misconception")
        st.space("small")
        chosen = row["option_text"] or "—"
        st.markdown(f"Chose **{ui.esc(chosen)}** &nbsp;{key_note}", unsafe_allow_html=True)

        if row["explanation"].strip():
            ui.html_block(f"<div class='rt-quote'>“{ui.esc(row['explanation'])}”</div>")
        else:
            ui.muted("No explanation written.")

        if row["predicted_label"]:
            st.markdown(f"{ui.tag('ml')} &nbsp;**{tx.title(row['predicted_label'])}**",
                        unsafe_allow_html=True)
            st.caption(f"Suggestion score {row['score']:.2f} — a ranking score, not a "
                       "probability. Low scores are the least certain.")
        else:
            st.markdown(f"{ui.tag('ml', 'No suggestion')} &nbsp;"
                        f"{tx.STATE_DESCRIPTIONS.get(row['state'], '')}",
                        unsafe_allow_html=True)
        if row["teacher_label"]:
            st.markdown(f"{ui.tag('teacher')} &nbsp;**{label_title(row['teacher_label'])}**",
                        unsafe_allow_html=True)

        default = row["teacher_label"] or row["predicted_label"]
        with st.form(f"form_{keys[i]}", border=False):
            st.radio("Category", LABEL_OPTIONS, key=f"cat_{keys[i]}",
                     index=LABEL_OPTIONS.index(default) if default in LABEL_OPTIONS else None,
                     format_func=label_title)
            st.form_submit_button("Save category", type="primary", on_click=save,
                                  args=(row.to_dict(), choice))

    with right, ui.card("review-queue"):
        st.markdown(f"**{choice}** · {th.plural(len(view), 'answer')}")
        source = ["Teacher" if t else ("Model" if p else "—")
                  for t, p in zip(view["teacher_label"], view["predicted_label"])]
        shown = [("Too little to judge" if (t or p) == tx.STATE_INSUFFICIENT
                  else tx.short_name(t or p)) if (t or p) else "—"
                 for t, p in zip(view["teacher_label"], view["predicted_label"])]
        st.dataframe(pd.DataFrame({
            "Student": view["student_id"],
            "Q": view["question_id"],
            "Category": shown,
            "From": source,
        }), hide_index=True, width="stretch", height=470,
            column_config={"Category": st.column_config.TextColumn(width="medium"),
                           "From": st.column_config.TextColumn(
                               help="Model = machine suggestion; Teacher = confirmed by you.")})
        st.selectbox("Jump to answer", keys, key="review_key",
                     format_func=lambda k: f"{keys.index(k) + 1}. {k.replace('|', ' · ')}")

if st.session_state.corrections:
    st.download_button(
        f"Download {th.plural(len(st.session_state.corrections), 'confirmed label')} (CSV)",
        corr.to_frame(st.session_state.corrections).to_csv(index=False),
        file_name="teacher_corrections.csv", mime="text/csv", icon=":material/download:")

# --- Advanced ----------------------------------------------------------------------
st.space("small")
with st.expander("Advanced: auto-accept threshold and how suggestions work"):
    model, metrics = state.get_model(), state.load_metrics()
    grid = [f"{t:.2f}" for t in mdl.THRESHOLD_GRID] + [REVIEW_ALL]
    current = state.threshold()
    st.session_state["threshold_widget"] = REVIEW_ALL if current > 1 else f"{current:.2f}"

    def _sync_threshold():
        v = st.session_state["threshold_widget"]
        st.session_state.threshold = 1.01 if v == REVIEW_ALL else float(v)

    st.select_slider("Auto-accept suggestions with a score of at least", grid,
                     key="threshold_widget", on_change=_sync_threshold,
                     help="Suggestions below this score go to the review list.")
    if metrics:
        table = pd.DataFrame(metrics["training"]["threshold_table_validation"])
        match = table[table["threshold"].round(2) == round(state.threshold(), 2)]
        if len(match):
            m = match.iloc[0]
            st.caption(f"On validation data this setting auto-accepts {th.pct(m.coverage)} of "
                       f"suggestions, {th.pct(m.accepted_accuracy)} of them correct.")
        else:
            st.caption("With this setting every suggestion goes to teacher review.")
        default = ("send all to review" if model.threshold > 1 else f"{model.threshold:.2f}")
        st.caption(f"Default: {default}. It was chosen on validation data with the rule "
                   f"“{metrics['training']['threshold_policy']}”"
                   + ("; no setting met it." if model.threshold > 1 else "."))
    st.markdown("**What the review statuses mean**")
    for s_key, (_, s_text) in STATUS_TEXT.items():
        st.markdown(f"- **{s_text}** — {tx.STATE_DESCRIPTIONS[s_key]}")
    st.markdown("**About the model.** A TF-IDF + logistic-regression classifier trained on "
                "synthetic, author-written explanations. Scores are uncalibrated and are used "
                "only to rank answers. Confirmed labels are written to "
                "`outputs/teacher_corrections.csv` with `used_for_training = False`.")
