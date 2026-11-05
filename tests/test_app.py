"""Headless smoke tests of the Streamlit app using streamlit.testing."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def at():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_starts_without_errors(at):
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert len(at.tabs) == 6
    assert at.title[0].value.startswith("Reteach")


def test_overview_metrics(at):
    labels = {m.label: m.value for m in at.metric}
    assert labels["Students"] == "60"


def test_min_support_slider_changes_groups(at):
    at.sidebar.slider[0].set_value(1).run()
    assert not at.exception and not at.error
    at.sidebar.slider[0].set_value(2).run()


def test_reset_demo(at):
    at.sidebar.button[0].click().run()
    assert not at.exception


def test_teacher_correction_flow(tmp_path, monkeypatch):
    store = tmp_path / "corrections.csv"
    monkeypatch.setenv("RETEACH_CORRECTIONS_PATH", str(store))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    form_selects = [s for s in at.selectbox if s.label == "Correct category"]
    form_selects[0].set_value("add_across")
    at.button(key="FormSubmitter:correction_form-Save correction").click().run()
    assert not at.exception
    assert len(at.session_state["corrections"]) == 1
    c = at.session_state["corrections"][0]
    assert c["corrected_label"] == "add_across" and c["source"] == "teacher_correction"
    assert store.exists()
    # reset clears session corrections but keeps the saved file
    at.sidebar.button[0].click().run()
    assert at.session_state["corrections"] == []
    assert store.exists()

