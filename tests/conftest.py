from pathlib import Path

import pytest

from reteach import validation as val

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def demo_questions():
    res = val.validate_questions(DATA / "demo" / "questions.csv")
    assert res.ok, res.errors
    return res.df


@pytest.fixture(scope="session")
def demo_responses(demo_questions):
    res = val.validate_responses(DATA / "demo" / "responses.csv", demo_questions)
    assert res.ok, res.errors
    return res.df


@pytest.fixture
def tiny_questions():
    """Two add questions, each offering add_across (A) and unscaled (C)."""
    csv = (
        "question_id,topic,prompt,option_id,option_text,is_correct,misconception_label,"
        "operation,operand_a,operand_b\n"
        "Q1,Add,1/2+1/3,A,2/5,FALSE,add_across,add,1/2,1/3\n"
        "Q1,Add,1/2+1/3,B,5/6,TRUE,,add,1/2,1/3\n"
        "Q1,Add,1/2+1/3,C,2/6,FALSE,unscaled_numerator,add,1/2,1/3\n"
        "Q2,Add,1/4+2/3,A,3/7,FALSE,add_across,add,1/4,2/3\n"
        "Q2,Add,1/4+2/3,B,11/12,TRUE,,add,1/4,2/3\n"
        "Q2,Add,1/4+2/3,C,3/12,FALSE,unscaled_numerator,add,1/4,2/3\n"
        "Q3,Cmp,1/3 or 1/5,A,1/3,TRUE,,compare_larger,1/3,1/5\n"
        "Q3,Cmp,1/3 or 1/5,B,1/5,FALSE,larger_denominator_bigger,compare_larger,1/3,1/5\n"
    )
    res = val.validate_questions(csv.encode())
    assert res.ok, res.errors
    return res.df


def responses_csv(rows):
    lines = ["student_id,question_id,selected_option,explanation"]
    lines += [",".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode()
