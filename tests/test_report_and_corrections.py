import pandas as pd

from reteach import analytics as an
from reteach import corrections as corr
from reteach import report
from reteach import taxonomy as tx


def test_report_export_smoke(demo_questions, demo_responses):
    scored = an.score_responses(demo_questions, demo_responses)
    classified = pd.DataFrame({"state": [tx.STATE_NEEDS_REVIEW, tx.STATE_BLANK]})
    md = report.build_report(demo_questions, scored, 2, classified)
    assert md.startswith("# Reteaching report")
    assert "Synthetic data" in md
    assert "Suggested topics to review" in md
    assert "not AI" in md
    assert "of 600 expected responses" in md


def test_corrections_have_provenance_and_are_not_used_for_training(tmp_path):
    row = {"student_id": "S001", "question_id": "Q8", "explanation": "add tops and bottoms",
           "predicted_label": tx.CORRECT, "score": 0.31, "state": tx.STATE_NEEDS_REVIEW}
    c = corr.make_correction(row, tx.ADD_ACROSS, dataset="demo")
    assert c["source"] == "teacher_correction" and c["used_for_training"] is False
    assert c["model_label"] == tx.CORRECT and c["corrected_label"] == tx.ADD_ACROSS
    path = corr.append_to_file([c], tmp_path / "c.csv")
    corr.append_to_file([c], path)
    df = pd.read_csv(path)
    assert len(df) == 2 and list(df.columns) == corr.COLUMNS


def test_correction_rejects_unknown_label():
    import pytest
    with pytest.raises(ValueError):
        corr.make_correction({"student_id": "S", "question_id": "Q", "explanation": ""}, "bogus",
                             dataset="demo")
