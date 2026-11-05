from conftest import DATA, responses_csv
from reteach import validation as val


def test_demo_files_are_valid(demo_questions, demo_responses):
    assert demo_questions["question_id"].nunique() == 10
    assert demo_responses["student_id"].nunique() == 60


def test_empty_file():
    res = val.validate_questions(b"")
    assert not res.ok and "empty" in res.errors[0]
    res = val.validate_questions(DATA / "invalid" / "empty.csv")
    assert not res.ok


def test_header_only_file(tiny_questions):
    res = val.validate_responses(b"student_id,question_id,selected_option\n", tiny_questions)
    assert not res.ok and "no data rows" in res.errors[0]


def test_missing_column():
    res = val.validate_questions(DATA / "invalid" / "questions_missing_column.csv")
    assert not res.ok and "is_correct" in res.errors[0]


def test_two_correct_options_rejected():
    res = val.validate_questions(DATA / "invalid" / "questions_two_correct.csv")
    assert any("Q7 has 2 options marked correct" in e for e in res.errors)


def test_wrong_answer_key_rejected():
    res = val.validate_questions(DATA / "invalid" / "questions_wrong_answer_key.csv")
    assert any(e.startswith("Question Q8") for e in res.errors)


def test_unknown_label_rejected(tiny_questions):
    csv = (b"question_id,topic,prompt,option_id,option_text,is_correct,misconception_label\n"
           b"Q1,T,P,A,1/2,TRUE,\nQ1,T,P,B,1/3,FALSE,made_up_label\n")
    res = val.validate_questions(csv)
    assert not res.ok and "Unknown misconception_label" in res.errors[0]


def test_question_without_operation_warns_but_passes():
    csv = (b"question_id,topic,prompt,option_id,option_text,is_correct,misconception_label\n"
           b"Q1,T,P,A,1/2,TRUE,\nQ1,T,P,B,1/3,false,\n")
    res = val.validate_questions(csv)
    assert res.ok and "not checked arithmetically" in res.warnings[0]


def test_messy_responses_are_cleaned_with_warnings(demo_questions):
    res = val.validate_responses(DATA / "invalid" / "responses_messy.csv", demo_questions)
    assert res.ok
    text = " ".join(res.warnings)
    assert "unknown question_id" in text
    assert "not an option" in text
    assert "duplicate" in text
    assert "no student_id" in text
    df = res.df
    # duplicate S001/Q7 -> last row (A) kept; blank answer kept as unanswered.
    assert df.loc[df["student_id"] == "S001", "selected_option"].tolist() == ["A"]
    assert df.loc[df["student_id"] == "S004", "selected_option"].tolist() == [""]
    assert set(df["student_id"]) == {"S001", "S004"}


def test_lowercase_option_is_accepted(tiny_questions):
    res = val.validate_responses(responses_csv([("S1", "Q1", "b", "")]), tiny_questions)
    assert res.ok and res.df["selected_option"].tolist() == ["B"]


def test_missing_explanation_column_warns(tiny_questions):
    res = val.validate_responses(b"student_id,question_id,selected_option\nS1,Q1,B\n",
                                 tiny_questions)
    assert res.ok and "explanation" in res.warnings[0]


def test_all_rows_invalid_is_an_error(tiny_questions):
    res = val.validate_responses(responses_csv([("S1", "Q9", "A", "")]), tiny_questions)
    assert not res.ok


def test_not_a_csv():
    res = val.validate_questions(b"\xff\xfe\x00garbage\x00\x00")
    assert not res.ok
