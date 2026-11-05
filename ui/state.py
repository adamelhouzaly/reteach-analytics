"""Session state, cached resources, and per-run derived data.

Streamlit reruns the whole script on every interaction. Values that must
survive page changes live in ``st.session_state`` under plain (non-widget)
keys; widgets copy their value into those keys through ``on_change``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from reteach import analytics as an
from reteach import grouping as gr
from reteach import model as mdl
from reteach import taxonomy as tx
from reteach import validation as val

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

DEMO_LABEL = "Demo class · Fraction addition quiz"
TO_REVIEW_STATES = (tx.STATE_NEEDS_REVIEW, tx.STATE_UNRECOGNIZED)


# --- Dataset -----------------------------------------------------------------
def load_demo() -> dict:
    q = val.validate_questions(DATA / "demo" / "questions.csv")
    r = val.validate_responses(DATA / "demo" / "responses.csv", q.df)
    return {"questions": q.df, "responses": r.df, "label": DEMO_LABEL, "synthetic": True,
            "warnings": q.warnings + r.warnings}


def _clear_working_state() -> None:
    st.session_state.corrections = []
    for key in ("review_key", "review_pos", "focus_misconception", "pending", "flash"):
        st.session_state.pop(key, None)


def reset_to_demo() -> None:
    st.session_state.dataset = load_demo()
    _clear_working_state()


def use_dataset(dataset: dict) -> None:
    st.session_state.dataset = dataset
    _clear_working_state()


def init() -> None:
    if "dataset" not in st.session_state:
        st.session_state.dataset = load_demo()
        st.session_state.corrections = []
    st.session_state.setdefault("min_support", an.DEFAULT_MIN_SUPPORT)
    st.session_state.setdefault("threshold", None)  # None = model's validated default


def dataset() -> dict:
    return st.session_state.dataset


def min_support() -> int:
    return int(st.session_state.min_support)


def flash(message: str, kind: str = "success") -> None:
    """Queue a message ('success' or 'warning') to show after the next rerun."""
    st.session_state.flash = (kind, message)


def pop_flash() -> tuple[str, str] | None:
    return st.session_state.pop("flash", None)


# --- Model / metrics ---------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the explanation classifier…")
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


@st.cache_data
def load_test_predictions() -> pd.DataFrame | None:
    path = ARTIFACTS / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


def threshold() -> float:
    t = st.session_state.threshold
    return get_model().threshold if t is None else float(t)


# --- Derived data ------------------------------------------------------------
@dataclass
class Context:
    questions: pd.DataFrame
    scored: pd.DataFrame
    cov: an.Coverage
    topics: pd.DataFrame
    questions_summary: pd.DataFrame
    misconceptions: pd.DataFrame
    suggestions: list[dict]
    patterns: pd.DataFrame
    groups: pd.DataFrame
    evidence: pd.DataFrame
    min_support: int


def context() -> Context:
    ds = dataset()
    q, ms = ds["questions"], min_support()
    scored = an.score_responses(q, ds["responses"])
    cov = an.coverage(q, scored)
    topics = an.topic_summary(scored)
    mis = an.misconception_summary(q, scored, ms)
    patterns = gr.student_patterns(q, scored, ms)
    return Context(
        questions=q, scored=scored, cov=cov, topics=topics,
        questions_summary=an.question_summary(q, scored), misconceptions=mis,
        suggestions=an.suggest_topics(topics, mis, cov.n_students, q),
        patterns=patterns, groups=gr.groups(patterns),
        evidence=gr.student_evidence(q, scored), min_support=ms,
    )


def teacher_labels() -> dict[tuple[str, str], str]:
    """Latest teacher-confirmed label per (student_id, question_id)."""
    return {(c["student_id"], c["question_id"]): c["corrected_label"]
            for c in st.session_state.corrections}


def classified(ctx: Context) -> pd.DataFrame:
    """All responses to questions that collected explanations, classified at the
    current threshold, with teacher-confirmed labels merged in."""
    scored = ctx.scored
    asked = set(scored.loc[scored["explanation"] != "", "question_id"])
    rows = scored[scored["question_id"].isin(asked)].reset_index(drop=True)
    if rows.empty:
        return rows
    model = get_model()
    at_t = mdl.TrainedModel(model.pipeline, threshold(), model.meta)
    c = mdl.classify(at_t, rows["explanation"])
    out = pd.concat([rows[["student_id", "question_id", "prompt", "option_text", "outcome",
                           "is_correct", "misconception_label"]], c], axis=1)
    labels = teacher_labels()
    out["teacher_label"] = [labels.get(k, "") for k in zip(out["student_id"],
                                                           out["question_id"])]
    out["to_review"] = out["state"].isin(TO_REVIEW_STATES) & (out["teacher_label"] == "")
    return out
