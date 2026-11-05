"""Denominators and grouping rules."""
import pytest

from conftest import responses_csv
from reteach import analytics as an
from reteach import grouping as gr
from reteach import taxonomy as tx
from reteach import validation as val


@pytest.fixture
def scored(tiny_questions):
    rows = [
        # S1: add_across on Q1 and Q2 (recurring at min_support=2), correct on Q3
        ("S1", "Q1", "A", "added tops and bottoms"), ("S1", "Q2", "A", ""), ("S1", "Q3", "A", ""),
        # S2: add_across once, blank on Q2, no row for Q3 (missing)
        ("S2", "Q1", "A", ""), ("S2", "Q2", "", ""),
        # S3: all correct
        ("S3", "Q1", "B", ""), ("S3", "Q2", "B", ""), ("S3", "Q3", "A", ""),
    ]
    res = val.validate_responses(responses_csv(rows), tiny_questions)
    assert res.ok
    return an.score_responses(tiny_questions, res.df)


def test_coverage_denominators(tiny_questions, scored):
    cov = an.coverage(tiny_questions, scored)
    assert cov.n_students == 3 and cov.n_questions == 3
    assert cov.expected == 9
    assert cov.answered == 7          # 8 rows, 1 blank
    assert cov.unanswered == 1
    assert cov.missing == 1           # S2 has no Q3 row
    assert cov.correct == 4           # S1-Q3, S3 x3
    assert cov.accuracy == pytest.approx(4 / 7)  # blanks/missing excluded


def test_blank_answer_is_not_wrong_or_a_misconception(scored):
    blank = scored[(scored["student_id"] == "S2") & (scored["question_id"] == "Q2")].iloc[0]
    assert blank["outcome"] == "unanswered"
    assert not blank["is_correct"]
    assert blank["misconception_label"] == ""


def test_question_summary_denominators(tiny_questions, scored):
    qs = an.question_summary(tiny_questions, scored).set_index("question_id")
    assert qs.loc["Q2", "answered"] == 2 and qs.loc["Q2", "unanswered"] == 1
    assert qs.loc["Q3", "missing"] == 1
    assert qs.loc["Q1", "accuracy"] == pytest.approx(1 / 3)


def test_misconception_summary_denominators(tiny_questions, scored):
    m = an.misconception_summary(tiny_questions, scored, 2).set_index("misconception")
    across = m.loc[tx.ADD_ACROSS]
    assert across["supporting_questions"] == ["Q1", "Q2"]
    assert across["opportunities"] == 5      # answered Q1/Q2 responses (S2-Q2 blank)
    assert across["selections"] == 3
    assert across["students_any"] == 2
    assert across["students_recurring"] == 1
    assert across["recurring_student_ids"] == ["S1"]
    ld = m.loc[tx.LARGER_DENOMINATOR]
    assert ld["opportunities"] == 2 and ld["selections"] == 0


def test_grouping_min_support(tiny_questions, scored):
    p2 = gr.student_patterns(tiny_questions, scored, 2).set_index("student_id")
    assert p2.loc["S1", "recurring"] == [tx.ADD_ACROSS]
    assert p2.loc["S2", "recurring"] == []                # one question = possible only
    assert p2.loc["S2", "possible"] == [tx.ADD_ACROSS]
    assert p2.loc["S2", "group"] == gr.NO_PATTERN
    assert "2 of 2 questions" in p2.loc["S1", "evidence"]
    p1 = gr.student_patterns(tiny_questions, scored, 1).set_index("student_id")
    assert p1.loc["S2", "recurring"] == [tx.ADD_ACROSS]
    p3 = gr.student_patterns(tiny_questions, scored, 3).set_index("student_id")
    assert p3.loc["S1", "recurring"] == []


def test_groups_collapse_students(tiny_questions, scored):
    g = gr.groups(gr.student_patterns(tiny_questions, scored, 1)).set_index("group")
    assert g.loc["Add across", "n_students"] == 2
    assert g.loc[gr.NO_PATTERN, "n_students"] == 1


def test_invalid_min_support(tiny_questions, scored):
    with pytest.raises(ValueError):
        gr.student_patterns(tiny_questions, scored, 0)


def test_suggestions_explain_reasons(tiny_questions, scored):
    topics = an.topic_summary(scored)
    mis = an.misconception_summary(tiny_questions, scored, 2)
    sugg = an.suggest_topics(topics, mis, 3, tiny_questions)
    add = next(s for s in sugg if s["topic"] == "Add")
    assert any("below the 70%" in r for r in add["reasons"])
    assert any("recurs for 1 of 3" in r for r in add["reasons"])


def test_demo_has_recurring_groups(demo_questions, demo_responses):
    scored = an.score_responses(demo_questions, demo_responses)
    g = gr.groups(gr.student_patterns(demo_questions, scored, 2))
    assert g["is_pattern"].sum() >= 4
    assert g["n_students"].sum() == 60
