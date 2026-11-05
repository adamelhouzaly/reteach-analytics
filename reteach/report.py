"""Export a concise, teacher-readable reteaching report (Markdown)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from . import analytics as an
from . import grouping as gr
from . import taxonomy as tx


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.0%}"


def build_report(
    questions: pd.DataFrame,
    scored: pd.DataFrame,
    min_support: int = an.DEFAULT_MIN_SUPPORT,
    classified: pd.DataFrame | None = None,
    dataset_label: str = "Synthetic demo class",
    is_synthetic: bool = True,
) -> str:
    cov = an.coverage(questions, scored)
    topics = an.topic_summary(scored)
    mis = an.misconception_summary(questions, scored, min_support)
    suggestions = an.suggest_topics(topics, mis, cov.n_students, questions)
    patterns = gr.student_patterns(questions, scored, min_support)
    groups = gr.groups(patterns)

    L = [f"# Reteaching report — {dataset_label}", "", f"Generated {date.today()}.", ""]
    if is_synthetic:
        L += ["> **Synthetic data.** All students and answers in this report are fictional.", ""]
    L += [
        "## Summary",
        "",
        f"- Students: {cov.n_students}; questions: {cov.n_questions}.",
        f"- Answered {cov.answered} of {cov.expected} expected responses "
        f"({_pct(cov.response_rate)}); {cov.unanswered} left blank, {cov.missing} missing.",
        f"- Accuracy on answered questions: {_pct(cov.accuracy)} ({cov.correct} of {cov.answered}). "
        "Blank and missing responses are excluded from accuracy.",
        "",
        "## Suggested topics to review",
        "",
    ]
    if not suggestions:
        L.append("No topic met the review rules (accuracy below "
                 f"{an.TOPIC_ACCURACY_THRESHOLD:.0%} or a misconception recurring for at least "
                 f"{an.MISCONCEPTION_SHARE_THRESHOLD:.0%} of students).")
    for s in suggestions:
        L.append(f"### {s['topic']} — accuracy {_pct(s['accuracy'])}")
        L += [f"- Why: {r}" for r in s["reasons"]]
        L += [f"- Suggested activity (authored guidance): {a}" for a in s["activities"]]
        L.append("")
    L += [
        "## Possible misconceptions (from the teacher-defined answer key, not AI)",
        "",
        f"A misconception is *recurring* for a student when they chose a matching distractor on at "
        f"least {min_support} distinct questions.",
        "",
        "| Misconception | Chosen / opportunities | Students (any) | Students (recurring) | "
        "Supporting questions |",
        "|---|---|---|---|---|",
    ]
    for r in mis.itertuples():
        L.append(f"| {r.name} | {r.selections} / {r.opportunities} ({_pct(r.selection_rate)}) | "
                 f"{r.students_any} of {r.students_with_opportunity} | {r.students_recurring} | "
                 f"{', '.join(r.supporting_questions)} |")
    L += ["", "## Student groups (anonymous IDs)", ""]
    for g in groups.itertuples():
        ids = ", ".join(g.student_ids)
        L.append(f"- **{g.group}** ({g.n_students}): {ids}")
    if classified is not None and len(classified):
        counts = classified["state"].value_counts()
        L += ["", "## Written explanations (ML classifier — suggestions only)", ""]
        L += [f"- {tx.display_name(state)}: {n}" for state, n in counts.items()]
        L.append("- Model categories are uncalibrated suggestions and require teacher review.")
    L += ["", "---", "Rule-based counts come from the answer key. Machine-learning outputs are "
          "labelled separately. This report does not measure learning outcomes."]
    return "\n".join(L) + "\n"
