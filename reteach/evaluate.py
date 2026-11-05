"""Train and evaluate the explanation classifier; write metrics and a report.

    python -m reteach.cli train      # fit on dev split, choose threshold, save model
    python -m reteach.cli evaluate   # evaluate saved model on held-out test split

The model that the app uses is the same model that is evaluated: it is fitted
on the development split only, so the test split stays untouched.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import RANDOM_SEED, __version__
from . import model as m
from . import taxonomy as tx
from .validation import validate_labeled

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "labeled" / "explanations.csv"
CONTEXTS = ROOT / "data" / "labeled" / "contexts.csv"
ARTIFACTS = ROOT / "artifacts"


def load_labeled() -> tuple[pd.DataFrame, pd.DataFrame]:
    res = validate_labeled(LABELED)
    if not res.ok:
        raise ValueError("Labeled data is invalid: " + "; ".join(res.errors))
    return res.df, pd.read_csv(CONTEXTS, dtype=str)


def train_model(seed: int = RANDOM_SEED) -> tuple[m.TrainedModel, dict]:
    labeled, contexts = load_labeled()
    dev, test = m.split_dataset(labeled, contexts, seed)
    C, cv_scores = m.select_C(dev, seed)
    proba = m.oof_probabilities(dev, C, seed)
    classes = sorted(dev["label"].unique())
    table = m.threshold_table(dev["label"], proba, classes)
    threshold = m.choose_threshold(table)
    oof_pred = np.asarray(classes)[proba.argmax(axis=1)]
    pipeline = m.train(dev, C)
    info = {
        "C": C,
        "cv_macro_f1_by_C": cv_scores,
        "validation_oof_accuracy": float(accuracy_score(dev["label"], oof_pred)),
        "validation_oof_macro_f1": float(f1_score(dev["label"], oof_pred, average="macro")),
        "threshold_table_validation": table.to_dict("records"),
        "threshold_policy": f"lowest threshold with validation accepted-accuracy >= "
                            f"{m.TARGET_ACCEPTED_ACCURACY:.0%}",
    }
    meta = {"version": __version__, "seed": seed, "trained_on": "dev split only",
            "n_train": len(dev), "C": C, "trained": str(date.today())}
    trained = m.TrainedModel(pipeline, threshold, meta)
    m.save(trained)
    return trained, info


def _metrics(y_true, y_pred, labels) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
    }


def run_evaluation(seed: int = RANDOM_SEED, out_dir: Path = ARTIFACTS) -> dict:
    trained, train_info = train_model(seed)
    labeled, contexts = load_labeled()
    dev, test = m.split_dataset(labeled, contexts, seed)
    overlap = set(dev["group_id"]) & set(test["group_id"])
    assert not overlap, f"group leakage: {overlap}"
    labels = list(tx.TEXT_LABELS)

    y_pred = trained.pipeline.predict(test["explanation"])
    proba = trained.pipeline.predict_proba(test["explanation"])
    score = proba.max(axis=1)
    model_metrics = _metrics(test["label"], y_pred, labels)

    dummy = DummyClassifier(strategy="most_frequent").fit(dev["explanation"], dev["label"])
    dummy_pred = dummy.predict(test["explanation"])
    dummy_metrics = _metrics(test["label"], dummy_pred, labels)

    accepted = score >= trained.threshold
    n_acc = int(accepted.sum())
    selective = {
        "threshold": trained.threshold,
        "coverage": n_acc / len(test),
        "accepted": n_acc,
        "routed_to_review": int(len(test) - n_acc),
        "accepted_accuracy": float((y_pred[accepted] == test["label"].to_numpy()[accepted]).mean())
        if n_acc else None,
        "review_accuracy": float((y_pred[~accepted] == test["label"].to_numpy()[~accepted]).mean())
        if n_acc < len(test) else None,
    }

    preds = test[["example_id", "group_id", "question_id", "explanation", "label"]].copy()
    preds["predicted"] = y_pred
    preds["score"] = np.round(score, 3)
    preds["would_review"] = ~accepted
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    errors = preds[preds["label"] != preds["predicted"]].sort_values("score", ascending=False)

    def counts(df):
        return {lab: {"examples": int((df["label"] == lab).sum()),
                      "families": int(df.loc[df["label"] == lab, "group_id"].nunique())}
                for lab in labels}

    metrics = {
        "data_note": "ALL DATA IS SYNTHETIC: author-written template explanations, not real "
                     "student work. Results do not establish classroom effectiveness.",
        "seed": seed,
        "dataset": {
            "labeled_rows_in_file": len(labeled),
            "dev": {"examples": len(dev), "families": int(dev["group_id"].nunique()),
                    "by_label": counts(dev),
                    "contexts": sorted(dev["question_id"].unique())},
            "test": {"examples": len(test), "families": int(test["group_id"].nunique()),
                     "by_label": counts(test),
                     "contexts": sorted(test["question_id"].unique())},
            "group_overlap_dev_test": sorted(overlap),
        },
        "training": train_info,
        "test": {
            "model": model_metrics,
            "baseline_most_frequent": {**dummy_metrics,
                                       "predicted_class": str(dummy_pred[0])},
            "confusion_matrix": {"labels": labels,
                                 "matrix": confusion_matrix(test["label"], y_pred,
                                                            labels=labels).tolist()},
            "selective_prediction": selective,
            "n_errors": len(errors),
        },
        "calibration": "not evaluated; scores are treated as uncalibrated",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "evaluation_report.md").write_text(render_report(metrics, errors))
    return metrics


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render_report(metrics: dict, errors: pd.DataFrame) -> str:
    t = metrics["test"]
    d = metrics["dataset"]
    labels = t["confusion_matrix"]["labels"]
    lines = [
        "# Explanation classifier — evaluation report (auto-generated)",
        "",
        f"> {metrics['data_note']}",
        "",
        f"Seed {metrics['seed']}. Dev: {d['dev']['examples']} examples / "
        f"{d['dev']['families']} families. Test: {d['test']['examples']} examples / "
        f"{d['test']['families']} families (held-out families AND held-out question contexts "
        f"{', '.join(d['test']['contexts'])}). Family overlap dev/test: "
        f"{len(d['group_overlap_dev_test'])}.",
        "",
        "## Headline (test split)",
        "",
        "| Model | Accuracy | Macro F1 |",
        "|---|---|---|",
        f"| TF-IDF + logistic regression (C={metrics['training']['C']}) | "
        f"{_pct(t['model']['accuracy'])} | {t['model']['macro_f1']:.3f} |",
        f"| Most-frequent baseline (predicts `{t['baseline_most_frequent']['predicted_class']}`) | "
        f"{_pct(t['baseline_most_frequent']['accuracy'])} | "
        f"{t['baseline_most_frequent']['macro_f1']:.3f} |",
        "",
        "## Per-class (test split)",
        "",
        "| Class | Precision | Recall | F1 | Support (examples) | Test families |",
        "|---|---|---|---|---|---|",
    ]
    for lab in labels:
        pc = t["model"]["per_class"][lab]
        lines.append(f"| {lab} | {pc['precision']:.2f} | {pc['recall']:.2f} | {pc['f1']:.2f} | "
                     f"{pc['support']} | {d['test']['by_label'][lab]['families']} |")
    lines += ["", "## Confusion matrix (rows = true, columns = predicted)", "",
              "| true \\ pred | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for lab, row in zip(labels, t["confusion_matrix"]["matrix"]):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    s = t["selective_prediction"]
    lines += [
        "", "## Review threshold",
        "",
        f"Chosen on dev out-of-fold predictions ({metrics['training']['threshold_policy']}): "
        f"**{s['threshold']:.2f}**. On test: coverage {_pct(s['coverage'])} "
        f"({s['accepted']} auto-accepted, {s['routed_to_review']} routed to review); "
        f"accuracy on accepted {_pct(s['accepted_accuracy'])}; accuracy on routed "
        f"{_pct(s['review_accuracy'])}. Scores are uncalibrated.",
        "",
        "## Misclassified test examples (highest score first)",
        "",
        "| Explanation | True | Predicted | Score |",
        "|---|---|---|---|",
    ]
    for r in errors.itertuples():
        text = r.explanation.replace("|", "/")
        lines.append(f"| {text} | {r.label} | {r.predicted} | {r.score:.2f} |")
    if errors.empty:
        lines.append("| (none) | | | |")
    return "\n".join(lines) + "\n"
