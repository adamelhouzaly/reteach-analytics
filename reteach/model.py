"""Written-explanation classifier: TF-IDF + logistic regression.

Key ideas (see docs/learning_guide.md for a beginner-friendly explanation):

* The TF-IDF vectoriser lives *inside* a scikit-learn ``Pipeline`` so it is
  fitted only on training folds — never on validation or test text.
* Splits are **group-aware**: all paraphrases of one template family share a
  ``group_id`` and land in the same split.
* Test examples also use **held-out question contexts** (different numbers).
* The review threshold is chosen from *out-of-fold validation* predictions on
  the development set, never from the test set.
* ``predict_proba`` scores are treated as **uncalibrated** confidence scores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from . import RANDOM_SEED
from . import taxonomy as tx

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "explanation_classifier.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

TEST_FAMILY_FRACTION = 1 / 3
N_FOLDS = 4
C_GRID = (0.3, 1.0, 3.0, 10.0)
TARGET_ACCEPTED_ACCURACY = 0.90  # policy for the review threshold (see choose_threshold)
THRESHOLD_GRID = tuple(np.round(np.arange(0.20, 0.96, 0.05), 2))
MIN_WORDS = 3
_WORD = re.compile(r"[A-Za-z]+")


def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_SEED)),
    ])


# --- Splitting -------------------------------------------------------------
def split_dataset(labeled: pd.DataFrame, contexts: pd.DataFrame, seed: int = RANDOM_SEED,
                  test_fraction: float = TEST_FAMILY_FRACTION
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dev, test). Families are split per label (stratified by label);
    dev rows use dev contexts and test rows use held-out test contexts."""
    rng = np.random.default_rng(seed)
    ctx_split = contexts.set_index("context_id")["context_split"]
    df = labeled.assign(context_split=labeled["question_id"].map(ctx_split))
    if df["context_split"].isna().any():
        raise ValueError("Every labeled example's question_id must appear in contexts.csv")
    test_groups: set[str] = set()
    for _, g in df.groupby("label"):
        groups = sorted(g["group_id"].unique())
        n_test = max(1, round(len(groups) * test_fraction))
        if len(groups) - n_test < N_FOLDS:
            raise ValueError(f"Label {g['label'].iloc[0]!r} has too few families to split.")
        test_groups.update(rng.choice(groups, size=n_test, replace=False))
    in_test = df["group_id"].isin(test_groups)
    dev = df[~in_test & (df["context_split"] == "dev")].reset_index(drop=True)
    test = df[in_test & (df["context_split"] == "test")].reset_index(drop=True)
    return dev, test


def _cv(seed: int = RANDOM_SEED) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)


def select_C(dev: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[float, dict[str, float]]:
    """Pick regularisation strength C by group-aware CV macro-F1 on dev only."""
    scores = {}
    for C in C_GRID:
        pred = cross_val_predict(build_pipeline(C), dev["explanation"], dev["label"],
                                 groups=dev["group_id"], cv=_cv(seed))
        scores[str(C)] = float(f1_score(dev["label"], pred, average="macro"))
    best = max(C_GRID, key=lambda c: (scores[str(c)], -c))
    return best, scores


def oof_probabilities(dev: pd.DataFrame, C: float, seed: int = RANDOM_SEED) -> np.ndarray:
    return cross_val_predict(build_pipeline(C), dev["explanation"], dev["label"],
                             groups=dev["group_id"], cv=_cv(seed), method="predict_proba")


def threshold_table(y_true, proba: np.ndarray, classes) -> pd.DataFrame:
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    score = proba.max(axis=1)
    y = np.asarray(y_true)
    rows = []
    for t in THRESHOLD_GRID:
        accepted = score >= t
        n = int(accepted.sum())
        rows.append({
            "threshold": float(t),
            "coverage": n / len(y),
            "accepted": n,
            "accepted_accuracy": float((pred[accepted] == y[accepted]).mean()) if n else None,
        })
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame, target: float = TARGET_ACCEPTED_ACCURACY) -> float:
    """Lowest threshold whose validation accepted-accuracy meets ``target``.
    Lowest = auto-accepts as many predictions as possible while meeting the
    policy. If nothing meets it, everything is routed to review (1.01)."""
    ok = table[table["accepted_accuracy"].fillna(0) >= target]
    return float(ok["threshold"].min()) if len(ok) else 1.01


# --- Training / persistence -----------------------------------------------
@dataclass
class TrainedModel:
    pipeline: Pipeline
    threshold: float
    meta: dict


def train(dev: pd.DataFrame, C: float) -> Pipeline:
    return build_pipeline(C).fit(dev["explanation"], dev["label"])


def save(model: TrainedModel, path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, path)
    meta_path.write_text(json.dumps({**model.meta, "threshold": model.threshold}, indent=2))


def load_local(path: Path = MODEL_PATH, meta_path: Path = META_PATH) -> TrainedModel | None:
    """Load the model THIS project trained locally. Only ever called with the
    fixed project path — the app never loads uploaded pickle/joblib files."""
    if not path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return TrainedModel(joblib.load(path), float(meta["threshold"]), meta)


# --- Inference -------------------------------------------------------------
def classify(model: TrainedModel, texts) -> pd.DataFrame:
    """Classify explanations into a label + handling state.

    Rules run first so weak inputs are never forced into a category:
    blank -> insufficient_text (< 3 words) -> unrecognized (no known words)
    -> model score below threshold = needs_review, else auto_accepted.
    """
    texts = pd.Series(list(texts), dtype=object).fillna("").astype(str)
    out = pd.DataFrame({"explanation": texts, "state": "", "predicted_label": "",
                        "score": np.nan})
    n_words = texts.map(lambda t: len(_WORD.findall(t)))
    out.loc[texts.str.strip() == "", "state"] = tx.STATE_BLANK
    out.loc[(out["state"] == "") & (n_words < MIN_WORDS), "state"] = tx.STATE_INSUFFICIENT
    todo = out.index[out["state"] == ""]
    if len(todo):
        X = model.pipeline.named_steps["tfidf"].transform(texts[todo])
        known = np.asarray(X.getnnz(axis=1)) > 0
        out.loc[todo[~known], "state"] = tx.STATE_UNRECOGNIZED
        idx = todo[known]
        if len(idx):
            proba = model.pipeline.predict_proba(texts[idx])
            classes = model.pipeline.classes_
            out.loc[idx, "predicted_label"] = classes[proba.argmax(axis=1)]
            out.loc[idx, "score"] = proba.max(axis=1)
            out.loc[idx, "state"] = np.where(proba.max(axis=1) >= model.threshold,
                                             tx.STATE_AUTO, tx.STATE_NEEDS_REVIEW)
    return out
