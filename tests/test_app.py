"""Headless tests of the Streamlit app (streamlit.testing AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
VIEWS = ["views/overview.py", "views/misconceptions.py", "views/groups.py", "views/review.py",
         "views/upload.py", "views/evaluation.py"]
DATA = Path(__file__).resolve().parent.parent / "data"


def new_app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    if page:
        at.switch_page(page).run()
    return at


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture(autouse=True)
def corrections_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RETEACH_DEBUG", "1")  # surface real exceptions in tests
    store = tmp_path / "corrections.csv"
    monkeypatch.setenv("RETEACH_CORRECTIONS_PATH", str(store))
    return store


@pytest.mark.parametrize("page", VIEWS)
def test_every_page_renders(page):
    assert_clean(new_app(page))


def test_overview_summary_cards():
    at = new_app()
    assert_clean(at)
    cards = {m.label: m.value for m in at.metric}
    assert cards["Students"] == "60"
    assert cards["Class accuracy"] == "59%"
    assert cards["Response coverage"] == "96%"
    assert int(cards["Answers to review"]) > 0
    assert any("Class overview" in h.value for h in at.markdown)


def test_reteach_action_opens_misconception_evidence():
    at = new_app()
    at.button(key="goto_mis").click().run()
    assert_clean(at)
    assert any("Selected from Overview" in m.value for m in at.markdown)


def test_open_review_queue_button():
    at = new_app()
    at.button(key="open_review").click().run()
    assert_clean(at)
    assert any(f.label == "Show" for f in at.segmented_control)


def test_min_support_setting_persists_across_pages():
    at = new_app("views/groups.py")
    at.segmented_control(key="min_support_widget_groups").set_value(4).run()
    assert_clean(at)
    assert at.session_state["min_support"] == 4
    at.switch_page("views/misconceptions.py").run()
    assert_clean(at)
    assert at.segmented_control(key="min_support_widget_mis").value == 4
    # threshold above available questions explains itself
    assert any("no student can reach" in i.value for i in at.info)


def test_teacher_correction_flow(corrections_store):
    at = new_app("views/review.py")
    key = at.session_state["review_key"]
    at.radio(key=f"cat_{key}").set_value("add_across")
    at.button(key=f"FormSubmitter:form_{key}-Save category").click().run()
    assert_clean(at)
    assert len(at.session_state["corrections"]) == 1
    c = at.session_state["corrections"][0]
    assert c["corrected_label"] == "add_across" and c["source"] == "teacher_correction"
    assert c["used_for_training"] is False
    assert corrections_store.exists()
    assert at.session_state["review_key"] != key          # moved on to the next answer
    # the queue keeps a stable order: the next answer is now first in the list
    assert any("Answer 1 of" in m.value for m in at.markdown)
    assert any("Saved" in s.value for s in at.success)
    # the confirmed answer appears under the Confirmed filter
    at.segmented_control(key="review_filter").set_value("Confirmed").run()
    assert_clean(at)
    assert at.session_state["review_key"] == key


def test_reset_to_demo_clears_session_labels(corrections_store):
    at = new_app("views/review.py")
    key = at.session_state["review_key"]
    at.radio(key=f"cat_{key}").set_value("correct_reasoning")
    at.button(key=f"FormSubmitter:form_{key}-Save category").click().run()
    at.switch_page("views/upload.py").run()
    at.button(key="reset_demo").click().run()
    assert_clean(at)
    assert at.session_state["corrections"] == []
    assert corrections_store.exists()                     # saved file is kept


def test_upload_messy_file_shows_feedback_and_can_be_used():
    at = new_app("views/upload.py")
    at.file_uploader(key="r_upload").upload(
        "responses_messy.csv", (DATA / "invalid" / "responses_messy.csv").read_bytes(),
        "text/csv").run()
    at.button(key="validate").click().run()
    assert_clean(at)
    text = " ".join(w.value for w in at.warning)
    assert "unknown question_id" in text and "duplicate" in text
    at.button(key="use_upload").click().run()
    assert_clean(at)
    assert at.session_state["dataset"]["synthetic"] is False
    assert {m.label: m.value for m in at.metric}["Students"] == "2"


def test_upload_invalid_questions_shows_error():
    at = new_app("views/upload.py")
    at.file_uploader(key="q_upload").upload(
        "questions_two_correct.csv",
        (DATA / "invalid" / "questions_two_correct.csv").read_bytes(), "text/csv")
    at.file_uploader(key="r_upload").upload(
        "responses.csv", (DATA / "demo" / "responses.csv").read_bytes(), "text/csv").run()
    at.checkbox(key="use_demo_q").uncheck().run()
    at.button(key="validate").click().run()
    assert not at.exception
    assert any("exactly 1 is required" in e.value for e in at.error)
