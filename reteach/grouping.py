"""Group students by *recurring* misconception patterns.

A single wrong answer is evidence of a *possible* misconception, not a
diagnosis. A misconception counts as "recurring" for a student only when they
chose a distractor mapped to it on at least ``min_support`` *distinct*
questions (default 2). Students are then grouped by the set of misconceptions
that recur for them. Everything here is rule-based on the teacher's answer
key — there is no machine learning in this module.
"""

from __future__ import annotations

import pandas as pd

from . import taxonomy as tx
from .analytics import DEFAULT_MIN_SUPPORT

NO_PATTERN = "No recurring pattern"
# Kept separate from NO_PATTERN: "no recurring pattern" does not mean "no difficulty".
ALL_CORRECT = "All answers correct"


def student_evidence(questions: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (student, misconception) with evidence and opportunity counts.

    * supporting_questions: distinct questions where the student picked a
      distractor mapped to the misconception.
    * opportunities: distinct questions the student *answered* that offered
      such a distractor.
    """
    answered = scored[scored["answered"]]
    students = sorted(scored["student_id"].unique())
    rows = []
    for label in tx.MISCONCEPTIONS:
        offering = set(questions.loc[questions["misconception_label"] == label, "question_id"])
        for sid in students:
            s = answered[answered["student_id"] == sid]
            opp = s[s["question_id"].isin(offering)]
            hits = sorted(set(opp.loc[opp["misconception_label"] == label, "question_id"]))
            rows.append(
                {
                    "student_id": sid,
                    "misconception": label,
                    "n_supporting": len(hits),
                    "supporting_questions": hits,
                    "opportunities": opp["question_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def evidence_status(n_supporting: int, min_support: int) -> str:
    if n_supporting == 0:
        return "none"
    if n_supporting >= min_support:
        return "recurring"
    return "possible (insufficient evidence)"


def student_patterns(
    questions: pd.DataFrame, scored: pd.DataFrame, min_support: int = DEFAULT_MIN_SUPPORT
) -> pd.DataFrame:
    """One row per student summarising their recurring and possible patterns."""
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    ev = student_evidence(questions, scored)
    answered = scored[scored["answered"]]
    rows = []
    for sid, s in ev.groupby("student_id", sort=True):
        recurring = s[s["n_supporting"] >= min_support]
        possible = s[(s["n_supporting"] > 0) & (s["n_supporting"] < min_support)]
        a = answered[answered["student_id"] == sid]
        key = tuple(sorted(recurring["misconception"]))
        if key:
            group = " + ".join(tx.short_name(k) for k in key)
        elif len(a) and int(a["is_correct"].sum()) == len(a):
            group = ALL_CORRECT
        else:
            group = NO_PATTERN
        rows.append(
            {
                "student_id": sid,
                "group": group,
                "recurring": list(key),
                "possible": sorted(possible["misconception"]),
                "evidence": "; ".join(
                    f"{tx.short_name(r.misconception)}: {r.n_supporting} of "
                    f"{r.opportunities} questions ({', '.join(r.supporting_questions)})"
                    for r in s[s["n_supporting"] > 0].itertuples()
                ),
                "answered": len(a),
                "correct": int(a["is_correct"].sum()),
                "unmapped_errors": int((a["outcome"] == "unmapped_error").sum()),
            }
        )
    return pd.DataFrame(rows)


def groups(patterns: pd.DataFrame) -> pd.DataFrame:
    """Collapse student patterns into anonymous groups."""
    if patterns.empty:
        return pd.DataFrame(columns=["group", "n_students", "student_ids", "misconceptions"])
    out = (
        patterns.groupby("group")
        .agg(
            n_students=("student_id", "size"),
            student_ids=("student_id", list),
            misconceptions=("recurring", "first"),
        )
        .reset_index()
    )
    out["is_pattern"] = ~out["group"].isin([NO_PATTERN, ALL_CORRECT])
    out["order"] = out["group"].map({NO_PATTERN: 1, ALL_CORRECT: 2}).fillna(0)
    return (out.sort_values(["order", "n_students"], ascending=[True, False])
            .drop(columns="order").reset_index(drop=True))
