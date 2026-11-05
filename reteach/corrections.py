"""Teacher corrections store.

Corrections are saved to their OWN file with provenance columns. They are
NEVER used to retrain automatically. A future, reviewed retraining workflow
would: (1) export this file, (2) have a second reviewer check each row,
(3) assign new group_ids so corrected items stay grouped, (4) add them to the
labeled dataset with source="teacher_correction", and (5) re-run the full
evaluation and compare against the saved metrics before replacing the model.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import taxonomy as tx

ROOT = Path(__file__).resolve().parent.parent
CORRECTIONS_PATH = ROOT / "outputs" / "teacher_corrections.csv"


def corrections_path() -> Path:
    """Default store; RETEACH_CORRECTIONS_PATH overrides it (used by tests)."""
    return Path(os.environ.get("RETEACH_CORRECTIONS_PATH", CORRECTIONS_PATH))


COLUMNS = ["correction_id", "created_utc", "dataset", "student_id", "question_id",
           "explanation", "model_label", "model_score", "model_state", "corrected_label",
           "source", "reviewer", "used_for_training"]


def make_correction(row: dict, corrected_label: str, dataset: str,
                    reviewer: str = "demo_teacher") -> dict:
    if corrected_label not in tx.TEXT_LABELS and corrected_label != tx.STATE_INSUFFICIENT:
        raise ValueError(f"Unknown label {corrected_label!r}")
    now = datetime.now(timezone.utc)
    score = row.get("score")
    return {
        "correction_id": f"{row['student_id']}-{row['question_id']}-{now:%Y%m%d%H%M%S%f}",
        "created_utc": now.isoformat(timespec="seconds"),
        "dataset": dataset,
        "student_id": row["student_id"],
        "question_id": row["question_id"],
        "explanation": row["explanation"],
        "model_label": row.get("predicted_label", ""),
        "model_score": None if score is None or pd.isna(score) else round(float(score), 3),
        "model_state": row.get("state", ""),
        "corrected_label": corrected_label,
        "source": "teacher_correction",
        "reviewer": reviewer,
        "used_for_training": False,
    }


def to_frame(corrections: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(corrections, columns=COLUMNS)


def append_to_file(corrections: list[dict], path: Path | None = None) -> Path:
    path = Path(path) if path else corrections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = to_frame(corrections)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
