"""Model evaluation: how well the explanation classifier did on held-out synthetic data."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reteach import taxonomy as tx
from ui import components as ui
from ui import state
from ui import theme as th

ui.page_header("Model evaluation",
               "How the written-explanation classifier performed on held-out test data.")
metrics = state.load_metrics()
if metrics is None:
    ui.empty_state("No evaluation results found.",
                   "Run `python -m reteach.cli evaluate` to create artifacts/metrics.json.")
    st.stop()

t, d = metrics["test"], metrics["dataset"]
model_m, base_m = t["model"], t["baseline_most_frequent"]
sel = t["selective_prediction"]

st.warning(
    "**These results come from synthetic data.** The labeled explanations were written by the "
    "project author from templates — not by students, and not labeled by experts. They show "
    "how the model separates these phrasings; they do not show accuracy on real student "
    "writing or any effect in the classroom.", icon=":material/science:")

# --- Headline cards --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1, ui.card("ev-acc"):
    st.metric("Test accuracy", th.pct(model_m["accuracy"]),
              help=f"Baseline (always the most common class): {th.pct(base_m['accuracy'])}.")
with c2, ui.card("ev-f1"):
    st.metric("Macro F1", f"{model_m['macro_f1']:.2f}",
              help=f"Average F1 across the 5 classes. Baseline: {base_m['macro_f1']:.2f}.")
with c3, ui.card("ev-size"):
    st.metric("Test set", f"{d['test']['examples']} answers",
              help=f"From {d['test']['families']} template families and question contexts the "
                   "model never saw during training.")
with c4, ui.card("ev-cov"):
    st.metric("Auto-accepted", th.pct(sel["coverage"]),
              help="Share of test suggestions that would skip teacher review at the default "
                   "threshold.")

st.space("small")
left, right = st.columns(2, gap="medium")
with left, ui.card("ev-baseline"):
    st.markdown("### Model vs. baseline")
    names = ["Accuracy", "Macro F1"]
    fig = go.Figure()
    fig.add_bar(name="TF-IDF + logistic regression", y=names,
                x=[model_m["accuracy"], model_m["macro_f1"]], orientation="h",
                marker=dict(color=th.TEAL, cornerradius=4),
                text=[f"{model_m['accuracy']:.0%}", f"{model_m['macro_f1']:.2f}"],
                textposition="outside", cliponaxis=False, hovertemplate="%{x:.3f}<extra></extra>")
    fig.add_bar(name="Most-frequent baseline", y=names,
                x=[base_m["accuracy"], base_m["macro_f1"]], orientation="h",
                marker=dict(color=th.SLATE, cornerradius=4),
                text=[f"{base_m['accuracy']:.0%}", f"{base_m['macro_f1']:.2f}"],
                textposition="outside", cliponaxis=False, hovertemplate="%{x:.3f}<extra></extra>")
    th.style_bar_chart(fig, 230, x_max=1.12)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1)
    th.legend_top(fig)
    st.plotly_chart(fig, config=th.PLOTLY_CONFIG, width="stretch")
    st.caption(f"The baseline always predicts “{tx.title(base_m['predicted_class'])}”.")

with right, ui.card("ev-confusion"):
    st.markdown("### Where the model gets confused")
    labels = t["confusion_matrix"]["labels"]
    short = [th.wrap_label(tx.short_name(x), 12) for x in labels]
    z = t["confusion_matrix"]["matrix"]
    peak = max(max(r) for r in z) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=short, y=short, colorscale=th.SEQ_TEAL, showscale=False, xgap=2, ygap=2,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>%{z} answers<extra></extra>"))
    # Cell counts: dark text on light cells, white text on dark cells.
    for yi, r in enumerate(z):
        for xi, v in enumerate(r):
            fig.add_annotation(x=short[xi], y=short[yi], text=str(v), showarrow=False,
                               font=dict(size=13, color="#ffffff" if v > peak * 0.55
                                         else th.NAVY))
    fig.update_layout(height=330, margin=dict(l=4, r=4, t=4, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", size=11, color=th.NAVY))
    fig.update_yaxes(autorange="reversed", title=dict(text="True category", font_size=12),
                     fixedrange=True)
    fig.update_xaxes(title=dict(text="Predicted category", font_size=12), fixedrange=True,
                     tickangle=0)
    st.plotly_chart(fig, config=th.PLOTLY_CONFIG, width="stretch")
    st.caption("Diagonal = correct. The biggest confusion: “unscaled numerator” explanations "
               "read like correct reasoning, because the difference is arithmetic, not wording.")

st.space("small")
with ui.card("ev-classes"):
    st.markdown("### Results by category")
    pc = pd.DataFrame(model_m["per_class"]).T.reset_index(names="label")
    st.dataframe(pd.DataFrame({
        "Category": pc["label"].map(tx.title),
        "Precision": pc["precision"].map(th.pct_int),
        "Recall": pc["recall"].map(th.pct_int),
        "F1": pc["f1"].map(lambda v: round(v, 2)),
        "Test answers": pc["support"].astype(int),
        "Test families": pc["label"].map(lambda x: d["test"]["by_label"][x]["families"]),
    }), hide_index=True, width="stretch", column_config={
        "Category": st.column_config.TextColumn(width="large"),
        "Precision": st.column_config.NumberColumn(format="%d%%"),
        "Recall": st.column_config.NumberColumn(format="%d%%"),
        "F1": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption("Precision: when the model picks this category, how often it is right. "
               "Recall: how many of the true answers in this category it finds.")

with st.expander("Review threshold"):
    st.markdown(
        f"The threshold was chosen on validation data (never test data) with the rule "
        f"“{metrics['training']['threshold_policy']}”. "
        + ("No setting met that rule, so **every suggestion goes to teacher review** by "
           "default." if sel["threshold"] > 1 else f"Chosen: {sel['threshold']:.2f}.")
        + f" On test data: {th.pct(sel['coverage'])} auto-accepted; suggestions sent to review "
          f"were {th.pct(sel['review_accuracy'])} correct. Scores are uncalibrated ranking "
          "scores, not probabilities.")
    table = pd.DataFrame(metrics["training"]["threshold_table_validation"])
    st.dataframe(pd.DataFrame({
        "Score at least": table["threshold"],
        "Auto-accepted": table["coverage"].map(th.pct_int),
        "Answers": table["accepted"],
        "Correct among accepted": table["accepted_accuracy"].map(th.pct_int),
    }), hide_index=True, width="stretch", column_config={
        "Score at least": st.column_config.NumberColumn(format="%.2f"),
        "Auto-accepted": st.column_config.NumberColumn(format="%d%%"),
        "Correct among accepted": st.column_config.NumberColumn(format="%d%%"),
    })

with st.expander("Examples the model got wrong"):
    preds = state.load_test_predictions()
    if preds is not None:
        err = preds[preds["label"] != preds["predicted"]].head(15)
        st.dataframe(pd.DataFrame({
            "Explanation": err["explanation"],
            "True category": err["label"].map(tx.short_name),
            "Model said": err["predicted"].map(tx.short_name),
            "Score": err["score"],
        }), hide_index=True, width="stretch", column_config={
            "Explanation": st.column_config.TextColumn(width="large"),
            "Score": st.column_config.NumberColumn(format="%.2f")})

with st.expander("Data and limitations"):
    st.markdown(
        f"- **Training:** {d['dev']['examples']} explanations from {d['dev']['families']} "
        f"template families. **Test:** {d['test']['examples']} from {d['test']['families']} "
        f"different families, using different numbers ({', '.join(d['test']['contexts'])}).\n"
        "- Rewordings of one idea always stay in the same split, so the test checks new "
        "wording rather than memorised sentences.\n"
        "- The text is cleaner than real student writing; real performance is unknown and "
        "would need real, consented, expert-labeled data.\n"
        "- Scores are not calibrated probabilities; calibration was not evaluated.\n"
        "- Full write-up: `docs/evaluation.md` and `docs/model_card.md`.")
