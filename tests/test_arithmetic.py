"""Answer keys are checked with Python's fractions module."""
from fractions import Fraction

import pytest

from reteach import arithmetic as ar
from reteach import datagen
from reteach import taxonomy as tx


def test_parse_keeps_unreduced_form():
    assert ar.parse_fraction("2/6") == (2, 6)
    assert ar.to_value("2/6") == Fraction(1, 3)
    assert ar.parse_fraction("They are equal") is None
    assert ar.parse_fraction("1/0") is None


@pytest.mark.parametrize("op,a,b,expected", [
    ("add", "1/2", "1/3", Fraction(5, 6)),
    ("add", "1/4", "2/3", Fraction(11, 12)),
    ("equivalent", "2/3", "9", Fraction(6, 9)),
    ("compare_larger", "2/7", "2/9", Fraction(2, 7)),
])
def test_correct_answer(op, a, b, expected):
    assert ar.correct_answer(op, a, b) == expected


def test_misconception_rules():
    m = ar.misconception_answers("add", "1/2", "1/3")
    assert m[tx.ADD_ACROSS] == {Fraction(2, 5)}
    assert Fraction(2, 6) in m[tx.UNSCALED_NUMERATOR]
    m = ar.misconception_answers("equivalent", "1/2", "8")
    assert m[tx.ADDITIVE_EQUIVALENCE] == {Fraction(7, 8)}
    assert m[tx.UNSCALED_NUMERATOR] == {Fraction(1, 8)}
    m = ar.misconception_answers("compare_larger", "1/3", "1/5")
    assert m[tx.LARGER_DENOMINATOR] == {Fraction(1, 5)}


def test_every_demo_question_has_exactly_one_correct_and_valid_mappings(demo_questions):
    for qid, q in demo_questions.groupby("question_id"):
        assert q["is_correct"].sum() == 1, qid
        problems = ar.check_question(q["operation"].iloc[0], q["operand_a"].iloc[0],
                                     q["operand_b"].iloc[0], q.to_dict("records"))
        assert problems == [], (qid, problems)


def test_every_misconception_has_at_least_two_supporting_questions(demo_questions):
    for label in tx.MISCONCEPTIONS:
        n = demo_questions.loc[demo_questions["misconception_label"] == label,
                               "question_id"].nunique()
        assert n >= 2, label


def test_wrong_mapping_is_detected():
    opts = [
        {"option_id": "A", "option_text": "5/6", "is_correct": True, "misconception_label": ""},
        {"option_id": "B", "option_text": "3/5", "is_correct": False,
         "misconception_label": tx.ADD_ACROSS},  # add-across would give 2/5
        {"option_id": "C", "option_text": "10/12", "is_correct": False,
         "misconception_label": ""},  # equals the correct answer
    ]
    problems = ar.check_question("add", "1/2", "1/3", opts)
    assert any("B" in p and "add_across" in p for p in problems)
    assert any("C" in p and "equals the correct answer" in p for p in problems)


def test_unlabelled_distractor_matching_a_rule_is_flagged():
    opts = [
        {"option_id": "A", "option_text": "5/6", "is_correct": True, "misconception_label": ""},
        {"option_id": "B", "option_text": "2/5", "is_correct": False, "misconception_label": ""},
    ]
    assert any("add_across" in p for p in ar.check_question("add", "1/2", "1/3", opts))


def test_quiz_definition_matches_written_file(demo_questions):
    assert demo_questions["question_id"].nunique() == len(datagen.QUIZ)
