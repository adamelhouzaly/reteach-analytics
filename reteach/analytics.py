"""Quiz analytics: accuracy, coverage, misconception counts, topic suggestions.

Denominator policy (tested in tests/test_analytics.py):

* **Expected responses** = number of students x number of questions. A
  "student" is anyone with at least one row in the responses file.
* **Answered** = a row exists AND selected_option is not blank.
* **Unanswered** = a row exists with a blank selected_option.
* **Missing** = no row at all for that student/question pair.
* **Accuracy** = correct / answered. Unanswered and missing responses are
  *excluded* from accuracy and reported separately as coverage, so a skipped
  question is never silently treated as a wrong answer or as a misconception.

Misconception counts come from the **teacher-defined answer key** (the
misconception_label on each distractor). They are rule-based, not AI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import taxonomy as tx

DEFAULT_MIN_SUPPORT = 2
TOPIC_ACCURACY_THRESHOLD = 0.70
MISCONCEPTION_SHARE_THRESHOLD = 0.20


def score_responses(questions: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """Join each response to its chosen option. One row per response row."""
    q_meta = questions.drop_duplicates("question_id")[["question_id", "topic", "prompt"]]
    opts = questions[["question_id", "option_id", "option_text", "is_correct", "misconception_label"]]
    df = responses.merge(q_meta, on="question_id", how="left")
    df = df.merge(
        opts, left_on=["question_id", "selected_option"], right_on=["question_id", "option_id"],
        how="left",
    ).drop(columns="option_id")
    df["answered"] = df["selected_option"] != ""
    df["is_correct"] = df["is_correct"].fillna(False).astype(bool) & df["answered"]
    df["misconception_label"] = df["misconception_label"].fillna("")
    df["option_text"] = df["option_text"].fillna("")
    df["outcome"] = "unanswered"
    df.loc[df["answered"] & df["is_correct"], "outcome"] = "correct"
    df.loc[df["answered"] & ~df["is_correct"] & (df["misconception_label"] != ""), "outcome"] = (
        "mapped_error"
    )
    df.loc[df["answered"] & ~df["is_correct"] & (df["misconception_label"] == ""), "outcome"] = (
        "unmapped_error"
    )
    return df


@dataclass
class Coverage:
    n_students: int
    n_questions: int
    expected: int
    answered: int
    unanswered: int
    missing: int
    correct: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.answered if self.answered else None

    @property
    def response_rate(self) -> float | None:
        return self.answered / self.expected if self.expected else None


def coverage(questions: pd.DataFrame, scored: pd.DataFrame) -> Coverage:
    n_students = scored["student_id"].nunique()
    n_questions = questions["question_id"].nunique()
    expected = n_students * n_questions
    answered = int(scored["answered"].sum())
    unanswered = int((~scored["answered"]).sum())
    return Coverage(
        n_students=n_students,
        n_questions=n_questions,
        expected=expected,
        answered=answered,
        unanswered=unanswered,
        missing=expected - len(scored),
        correct=int(scored["is_correct"].sum()),
    )


def question_summary(questions: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    n_students = scored["student_id"].nunique()
    rows = []
    for qid, q in questions.groupby("question_id", sort=False):
        s = scored[scored["question_id"] == qid]
        answered = s[s["answered"]]
        wrong = answered[~answered["is_correct"]]
        top_wrong = ""
        if len(wrong):
            counts = wrong["option_text"].value_counts()
            top_wrong = f"{counts.index[0]} ({counts.iloc[0]} of {len(answered)})"
        rows.append(
            {
                "question_id": qid,
                "topic": q["topic"].iloc[0],
                "prompt": q["prompt"].iloc[0],
                "correct_answer": q.loc[q["is_correct"], "option_text"].iloc[0],
                "answered": len(answered),
                "correct": int(answered["is_correct"].sum()),
                "accuracy": answered["is_correct"].mean() if len(answered) else None,
                "unanswered": int((~s["answered"]).sum()),
                "missing": n_students - len(s),
                "most_common_wrong": top_wrong,
            }
        )
    return pd.DataFrame(rows)


def topic_summary(scored: pd.DataFrame) -> pd.DataFrame:
    answered = scored[scored["answered"]]
    out = (
        answered.groupby("topic", sort=False)
        .agg(answered=("is_correct", "size"), correct=("is_correct", "sum"),
             questions=("question_id", "nunique"))
        .reset_index()
    )
    out["accuracy"] = out["correct"] / out["answered"]
    return out


def misconception_summary(
    questions: pd.DataFrame, scored: pd.DataFrame, min_support: int = DEFAULT_MIN_SUPPORT
) -> pd.DataFrame:
    """Per-misconception counts with explicit denominators.

    * supporting_questions: questions whose answer key has a distractor for it.
    * opportunities: answered responses to those questions.
    * selections: responses that chose a distractor mapped to it.
    * students_with_opportunity: students who answered >=1 supporting question.
    * students_any: students who chose a mapped distractor at least once.
    * students_recurring: students who chose it on >= min_support distinct questions.
    """
    answered = scored[scored["answered"]]
    rows = []
    for label in tx.MISCONCEPTIONS:
        qids = sorted(set(questions.loc[questions["misconception_label"] == label, "question_id"]))
        opp = answered[answered["question_id"].isin(qids)]
        sel = opp[opp["misconception_label"] == label]
        per_student = sel.groupby("student_id")["question_id"].nunique()
        rows.append(
            {
                "misconception": label,
                "name": tx.display_name(label),
                "supporting_questions": qids,
                "n_supporting_questions": len(qids),
                "opportunities": len(opp),
                "selections": len(sel),
                "selection_rate": len(sel) / len(opp) if len(opp) else None,
                "students_with_opportunity": opp["student_id"].nunique(),
                "students_any": int(per_student.size),
                "students_recurring": int((per_student >= min_support).sum()),
                "recurring_student_ids": sorted(per_student[per_student >= min_support].index),
            }
        )
    return pd.DataFrame(rows)


def unmapped_error_summary(scored: pd.DataFrame) -> dict:
    answered = scored[scored["answered"]]
    unmapped = answered[answered["outcome"] == "unmapped_error"]
    return {"count": len(unmapped), "answered": len(answered),
            "students": unmapped["student_id"].nunique()}


def suggest_topics(
    topics: pd.DataFrame,
    misconceptions: pd.DataFrame,
    n_students: int,
    questions: pd.DataFrame,
    accuracy_threshold: float = TOPIC_ACCURACY_THRESHOLD,
    share_threshold: float = MISCONCEPTION_SHARE_THRESHOLD,
) -> list[dict]:
    """Deterministic, explainable review suggestions.

    A topic is suggested when (a) its accuracy is below ``accuracy_threshold``
    or (b) a misconception linked to it recurs for at least ``share_threshold``
    of students. A misconception is linked to a topic when one of the topic's
    questions has a distractor mapped to it in the answer key. Each suggestion
    carries the reasons and the authored activity.
    """
    linked_topics = (questions[questions["misconception_label"] != ""]
                     .groupby("misconception_label")["topic"].agg(set).to_dict())
    suggestions = []
    for _, t in topics.iterrows():
        reasons = []
        if t["accuracy"] < accuracy_threshold:
            reasons.append(
                f"Accuracy is {t['accuracy']:.0%} ({int(t['correct'])} of {int(t['answered'])} "
                f"answered), below the {accuracy_threshold:.0%} review threshold."
            )
        linked = []
        for _, m in misconceptions.iterrows():
            cat = tx.CATEGORIES[m["misconception"]]
            if t["topic"] not in linked_topics.get(m["misconception"], set()) or not n_students:
                continue
            share = m["students_recurring"] / n_students
            if share >= share_threshold:
                reasons.append(
                    f"'{cat.name}' recurs for {m['students_recurring']} of {n_students} students "
                    f"({share:.0%})."
                )
                linked.append(cat)
        if reasons:
            suggestions.append(
                {
                    "topic": t["topic"],
                    "accuracy": t["accuracy"],
                    "reasons": reasons,
                    "activities": [c.review_activity for c in linked],
                    "linked_misconceptions": [c.label for c in linked],
                }
            )
    suggestions.sort(key=lambda s: s["accuracy"])
    return suggestions
