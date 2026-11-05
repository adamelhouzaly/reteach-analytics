"""Design tokens, global CSS, Plotly styling and number formatting.

Colours are mirrored in .streamlit/config.toml. Custom CSS only targets
classes we create ourselves: ``st-key-<key>`` classes, which Streamlit adds to
containers created with ``st.container(key=...)``, and our own ``rt-*`` spans.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY = "#1b2a41"
MUTED = "#526077"        # 6.3:1 on white
BORDER = "#dde3ea"
PAGE = "#f3f5f7"
TEAL = "#0f766e"         # primary accent
TEAL_LIGHT = "#8fd0c6"   # secondary series (paired with TEAL, always labelled)
TEAL_TINT = "#e8f5f2"
AMBER = "#b45309"        # attention text (5.0:1 on white)
AMBER_BAR = "#d97706"    # attention marks
AMBER_TINT = "#fff6e5"
RED = "#b42318"          # errors only
SLATE = "#9aa7b8"        # neutral marks (baseline, "no pattern")
INDIGO = "#3f3d9e"       # model-suggestion tag text
SEQ_TEAL = ["#f3faf8", "#d3eee9", "#a6dbd2", "#6fc0b3", "#3a9f91", "#0f766e", "#0b544f"]

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_CSS = f"""
<style>
  /* Layout width + comfortable top spacing */
  [data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 1.5rem; }}

  /* Cards: any st.container(key="card-...") */
  [class*="st-key-card"] {{
    background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.25rem; box-shadow: 0 1px 2px rgba(27, 42, 65, 0.04);
  }}
  [class*="st-key-card-kpi"] {{ min-height: 152px; min-width: 150px; }}
  /* Let metric labels wrap instead of truncating on narrow cards */
  [class*="st-key-card-kpi"] [data-testid="stMetricLabel"] * {{
    white-space: normal; overflow: visible; text-overflow: clip; }}
  [class*="st-key-card-priority"] {{ border-left: 4px solid {TEAL}; }}
  .rt-item {{ margin-bottom: 0.85rem; }}
  .rt-item-title {{ font-weight: 600; color: {NAVY}; }}
  [class*="st-key-card-attention"] {{ border-left: 4px solid {AMBER_BAR}; }}

  /* Small text styles */
  .rt-eyebrow {{ font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
                 text-transform: uppercase; color: {TEAL}; margin-bottom: 0.15rem; }}
  .rt-eyebrow.attention {{ color: {AMBER}; }}
  .rt-muted {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.45; }}
  .rt-big {{ font-size: 1.35rem; font-weight: 650; color: {NAVY}; line-height: 1.25; }}
  .rt-stat {{ font-size: 1.5rem; font-weight: 650; color: {NAVY}; line-height: 1.1; }}
  .rt-stat-label {{ font-size: 0.85rem; color: {MUTED}; margin-bottom: 0.75rem; }}
  .rt-quote {{ border-left: 3px solid {BORDER}; padding: 0.4rem 0 0.4rem 0.9rem;
               font-size: 1.08rem; color: {NAVY}; margin: 0.4rem 0 0.6rem 0; }}

  /* Tags (source of a label) */
  .rt-tag {{ display: inline-block; padding: 1px 9px; border-radius: 999px;
             font-size: 0.76rem; font-weight: 600; border: 1px solid transparent;
             white-space: nowrap; vertical-align: middle; }}
  .rt-tag.rule {{ background: {TEAL_TINT}; color: #0b544f; border-color: #c4e6df; }}
  .rt-tag.ml {{ background: #eeeefb; color: {INDIGO}; border-color: #d6d5f3; }}
  .rt-tag.teacher {{ background: {TEAL}; color: #ffffff; }}
  .rt-tag.attention {{ background: {AMBER_TINT}; color: {AMBER}; border-color: #f6dcae; }}
  .rt-tag.neutral {{ background: #f1f3f6; color: #3b4a5f; border-color: {BORDER}; }}

  /* Keep long words (student IDs, fractions) from overflowing on phones */
  [class*="st-key-card"] p {{ overflow-wrap: anywhere; }}
  @media (max-width: 640px) {{
    [data-testid="stMainBlockContainer"] {{ padding-left: 1rem; padding-right: 1rem; }}
    [class*="st-key-card"] {{ padding: 0.9rem 1rem; }}
  }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# --- Formatting --------------------------------------------------------------
def pct(x) -> str:
    """0.6347 -> '63%'. Missing -> an em dash."""
    return "—" if x is None or pd.isna(x) else f"{x:.0%}"


def pct_int(x):
    """For dataframe columns: 0.6347 -> 63 (displayed with a % format)."""
    return None if x is None or pd.isna(x) else round(float(x) * 100)


def plural(n: int, word: str, plural_word: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural_word or word + 's')}"


# --- Charts ------------------------------------------------------------------
def style_bar_chart(fig: go.Figure, height: int, x_title: str | None = None,
                    percent: bool = False, x_max: float | None = None) -> go.Figure:
    """Clean horizontal-bar styling: no y grid, faint x grid, readable labels."""
    fig.update_layout(
        height=height, margin=dict(l=4, r=16, t=8, b=4), bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=NAVY),
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font_size=13,
                        font_color=NAVY),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f3", zeroline=False, ticks="",
                     title=dict(text=x_title, font=dict(size=12, color=MUTED)),
                     tickfont=dict(color=MUTED, size=12),
                     tickformat=".0%" if percent else None,
                     range=[0, x_max] if x_max else None, fixedrange=True)
    fig.update_yaxes(showgrid=False, autorange="reversed", ticks="", title=None,
                     tickfont=dict(size=13, color=NAVY), automargin=True, fixedrange=True)
    return fig


def legend_top(fig: go.Figure) -> go.Figure:
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, yanchor="bottom",
                                                   x=0, title=None, font=dict(size=12)))
    fig.update_layout(margin=dict(t=36))
    return fig


def wrap_label(text: str, width: int = 24) -> str:
    """Break long category labels onto two lines for chart axes."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "<br>".join(lines)
