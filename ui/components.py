"""Small reusable UI building blocks."""

from __future__ import annotations

import html
import itertools

import streamlit as st

from ui import state

_ids = itertools.count()

DEMO_HELP = ("All students, answers and written explanations in the demo are fictional "
             "(synthetic). See Model evaluation for what this means for the results.")

TAG_TEXT = {
    "rule": ("rule", "Answer-key rule"),
    "ml": ("ml", "Model suggestion"),
    "teacher": ("teacher", "Teacher-confirmed"),
}


def esc(text) -> str:
    return html.escape(str(text))


def tag(kind: str, text: str | None = None, title: str | None = None) -> str:
    """Inline pill. kind: rule | ml | teacher | attention | neutral."""
    css, default = TAG_TEXT.get(kind, (kind, kind))
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="rt-tag {css}"{t}>{esc(text or default)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def card(name: str | None = None, variant: str = ""):
    """A white card. ``variant`` may be 'priority' or 'attention'."""
    key = f"card-{variant + '-' if variant else ''}{name or next(_ids)}"
    return st.container(key=key)


def page_header(title: str, subtitle: str | None = None) -> None:
    ds = state.dataset()
    st.markdown(f"# {title}")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"<span class='rt-muted'>{esc(ds['label'])}</span>",
                    unsafe_allow_html=True)
        if ds["synthetic"]:
            st.badge("Demo data", color="gray", help=DEMO_HELP)
        else:
            st.badge("Uploaded data", color="green")
    if subtitle:
        st.markdown(f"<div class='rt-muted' style='margin-top:0.25rem'>{esc(subtitle)}</div>",
                    unsafe_allow_html=True)
    msg = state.pop_flash()
    if msg:
        kind, text = msg
        (st.warning if kind == "warning" else st.success)(text)
    st.space("small")


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def stat(value, label: str, help_text: str | None = None) -> None:
    """A compact number + label (for use inside cards)."""
    t = f' title="{esc(help_text)}"' if help_text else ""
    html_block(f"<div{t}><div class='rt-stat'>{esc(value)}</div>"
               f"<div class='rt-stat-label'>{esc(label)}</div></div>")


def eyebrow(text: str, attention: bool = False) -> None:
    html_block(f"<div class='rt-eyebrow{' attention' if attention else ''}'>{esc(text)}</div>")


def muted(text: str) -> None:
    html_block(f"<div class='rt-muted'>{esc(text)}</div>")


def empty_state(message: str, hint: str | None = None) -> None:
    with card():
        st.markdown(f"**{message}**")
        if hint:
            muted(hint)


def min_support_control(where: str) -> int:
    """Shared control for the recurring-pattern threshold (persists across pages)."""
    key = f"min_support_widget_{where}"
    st.session_state[key] = state.min_support()

    def _sync():
        st.session_state.min_support = st.session_state[key]

    st.segmented_control(
        "Recurring pattern = same misconception on at least",
        options=[1, 2, 3, 4, 5], key=key, on_change=_sync, required=True,
        format_func=lambda n: f"{n} question" + ("s" if n > 1 else ""),
        help="One wrong answer is evidence, not a diagnosis. The default of 2 asks for the "
             "same misconception on two different questions. This setting applies on every "
             "page.")
    return state.min_support()


def go(page: str) -> None:
    """Navigate to another page by its key in app.PAGES (e.g. 'review')."""
    st.switch_page(st.session_state["_pages"][page])
