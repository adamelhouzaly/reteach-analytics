"""CSV schemas and input validation.

Validation never raises on bad user data. It returns a ``ValidationResult``
holding a cleaned DataFrame plus lists of *errors* (which block analysis) and
*warnings* (problems we worked around, e.g. dropped rows). The Streamlit app
shows these messages directly to the teacher.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

from . import arithmetic
from . import taxonomy as tx

# --- Schemas ---------------------------------------------------------------
QUESTION_REQUIRED = [
    "question_id",
    "topic",
    "prompt",
    "option_id",
    "option_text",
    "is_correct",
    "misconception_label",
]
# Optional: enables automatic arithmetic checking of the answer key.
QUESTION_OPTIONAL = ["operation", "operand_a", "operand_b"]

RESPONSE_REQUIRED = ["student_id", "question_id", "selected_option"]
RESPONSE_OPTIONAL = ["explanation"]

LABELED_REQUIRED = ["example_id", "question_id", "explanation", "label", "source", "group_id"]

SCHEMA_DOCS = {
    "questions": {
        "question_id": "Short ID shared by all options of one question, e.g. Q5.",
        "topic": "Topic name; identical for every row of the question.",
        "prompt": "Question text shown to students.",
        "option_id": "Option letter/ID, unique within the question (A, B, C...).",
        "option_text": "Option as shown, e.g. 5/6. Fractions are written a/b.",
        "is_correct": "TRUE for exactly one option per question, FALSE otherwise.",
        "misconception_label": "Teacher-defined misconception this distractor signals, "
        "or blank. Allowed: " + ", ".join(tx.MISCONCEPTIONS) + ".",
        "operation (optional)": "add | equivalent | compare_larger — enables arithmetic checks.",
        "operand_a (optional)": "First fraction, e.g. 1/2.",
        "operand_b (optional)": "Second fraction (add, compare_larger) or target denominator "
        "(equivalent).",
    },
    "responses": {
        "student_id": "Anonymous student ID (do not upload names).",
        "question_id": "Must match a question_id in the questions file.",
        "selected_option": "The option_id chosen. Leave blank if the student skipped it.",
        "explanation (optional)": "Short written explanation. May be blank.",
    },
}

_TRUE = {"true", "t", "1", "yes", "y"}
_FALSE = {"false", "f", "0", "no", "n", ""}
_MAX_LISTED = 5


@dataclass
class ValidationResult:
    df: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _n(count: int, word: str) -> str:
    """'1 row' / '3 rows'."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def _lines(index: pd.Index) -> str:
    return ("line " if len(index) == 1 else "lines ") + _rows(index)


def _rows(index: pd.Index) -> str:
    """Format DataFrame index values as 1-based CSV line numbers (header = line 1)."""
    lines = [str(i + 2) for i in list(index)[:_MAX_LISTED]]
    more = len(index) - len(lines)
    return ", ".join(lines) + (f" (+{more} more)" if more > 0 else "")


def read_csv(source) -> ValidationResult:
    """Read a CSV from a path, bytes, or file-like object as all-text columns."""
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        df = pd.read_csv(source, dtype=str, keep_default_na=False, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        return ValidationResult(pd.DataFrame(), errors=["The file is empty."])
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return ValidationResult(
            pd.DataFrame(), errors=[f"Could not read the file as UTF-8 CSV ({type(exc).__name__})."]
        )
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.apply(lambda col: col.str.strip())
    if df.empty:
        return ValidationResult(df, errors=["The file has a header row but no data rows."])
    return ValidationResult(df)


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def validate_questions(source) -> ValidationResult:
    """Validate a question/answer-key CSV (one row per answer option)."""
    res = read_csv(source)
    if not res.ok:
        return res
    df = res.df
    missing = _missing_columns(df, QUESTION_REQUIRED)
    if missing:
        res.errors.append(
            "Missing required column(s): " + ", ".join(missing) + ". Download the template "
            "to see the expected header."
        )
        return res
    for col in QUESTION_OPTIONAL:
        if col not in df.columns:
            df[col] = ""

    errors, warnings = res.errors, res.warnings
    df["option_id"] = df["option_id"].str.upper()
    for col in ["question_id", "topic", "prompt", "option_id", "option_text"]:
        blank = df.index[df[col] == ""]
        if len(blank):
            errors.append(f"Column '{col}' is blank on {_lines(blank)}.")

    lowered = df["is_correct"].str.lower()
    bad_bool = df.index[~lowered.isin(_TRUE | _FALSE)]
    if len(bad_bool):
        errors.append(
            f"'is_correct' must be TRUE or FALSE; check {_lines(bad_bool)}."
        )
    df["is_correct"] = lowered.isin(_TRUE)

    bad_label = df.index[~df["misconception_label"].isin([*tx.MISCONCEPTIONS, tx.UNMAPPED])]
    if len(bad_label):
        errors.append(
            f"Unknown misconception_label on {_lines(bad_label)}. Allowed values: "
            + ", ".join(tx.MISCONCEPTIONS)
            + " (or blank)."
        )

    dup = df.index[df.duplicated(["question_id", "option_id"], keep="first")]
    if len(dup):
        errors.append(f"Duplicate question_id + option_id on {_lines(dup)}.")

    if errors:
        return res

    for qid, q in df.groupby("question_id", sort=False):
        n_correct = int(q["is_correct"].sum())
        if n_correct != 1:
            errors.append(
                f"Question {qid} has {n_correct} options marked correct; exactly 1 is required."
            )
        if len(q) < 2:
            errors.append(f"Question {qid} has only {len(q)} option; at least 2 are required.")
        for col in ["topic", "prompt", "operation", "operand_a", "operand_b"]:
            if q[col].nunique() > 1:
                errors.append(f"Question {qid} has different '{col}' values across its rows.")
        op = q["operation"].iloc[0]
        if op == "":
            warnings.append(
                f"Question {qid} has no 'operation', so its answer key was not checked "
                "arithmetically."
            )
            continue
        if op not in arithmetic.OPERATIONS:
            errors.append(
                f"Question {qid}: unknown operation {op!r}. Use one of "
                + ", ".join(arithmetic.OPERATIONS)
                + " or leave blank."
            )
            continue
        problems = arithmetic.check_question(
            op, q["operand_a"].iloc[0], q["operand_b"].iloc[0], q.to_dict("records")
        )
        errors.extend(f"Question {qid}: {p}" for p in problems)
    return res


def validate_responses(source, questions: pd.DataFrame) -> ValidationResult:
    """Validate a response CSV against already-validated questions.

    Policy (also documented in the README):
    * blank selected_option  -> kept, counted as *unanswered* (not in accuracy).
    * unknown question_id    -> row dropped with a warning.
    * invalid selected_option-> row dropped with a warning.
    * duplicate student+question -> last row kept, warning.
    * missing student_id     -> row dropped with a warning.
    """
    res = read_csv(source)
    if not res.ok:
        return res
    df = res.df
    missing = _missing_columns(df, RESPONSE_REQUIRED)
    if missing:
        res.errors.append(
            "Missing required column(s): " + ", ".join(missing) + ". Download the template "
            "to see the expected header."
        )
        return res
    if "explanation" not in df.columns:
        df["explanation"] = ""
        res.warnings.append("No 'explanation' column — written-answer analysis is unavailable.")

    warnings = res.warnings
    df["selected_option"] = df["selected_option"].str.upper()
    keep = pd.Series(True, index=df.index)

    no_id = df.index[df["student_id"] == ""]
    if len(no_id):
        warnings.append(f"Dropped {_n(len(no_id), 'row')} with no student_id ({_lines(no_id)}).")
        keep[no_id] = False

    known_q = set(questions["question_id"])
    unknown = df.index[keep & ~df["question_id"].isin(known_q)]
    if len(unknown):
        ids = sorted(set(df.loc[unknown, "question_id"]))[:_MAX_LISTED]
        warnings.append(
            f"Dropped {_n(len(unknown), 'row')} with unknown question_id "
            f"{', '.join(ids)} ({_lines(unknown)})."
        )
        keep[unknown] = False

    valid_pairs = set(zip(questions["question_id"], questions["option_id"].str.upper()))
    pairs = list(zip(df["question_id"], df["selected_option"]))
    is_valid_choice = pd.Series([p in valid_pairs for p in pairs], index=df.index)
    invalid = df.index[keep & (df["selected_option"] != "") & ~is_valid_choice]
    if len(invalid):
        warnings.append(
            f"Dropped {_n(len(invalid), 'row')} whose selected_option is not an option for "
            f"that question ({_lines(invalid)})."
        )
        keep[invalid] = False

    df = df[keep]
    dup = df.duplicated(["student_id", "question_id"], keep="last")
    if dup.any():
        warnings.append(
            f"Found {_n(int(dup.sum()), 'duplicate student/question row')}; kept the last "
            f"one ({_lines(df.index[dup])})."
        )
        df = df[~dup]

    if df.empty:
        res.errors.append("No usable response rows remain after validation.")
    res.df = df.reset_index(drop=True)
    return res


def validate_labeled(source) -> ValidationResult:
    """Validate a labeled-explanation training CSV."""
    res = read_csv(source)
    if not res.ok:
        return res
    missing = _missing_columns(res.df, LABELED_REQUIRED)
    if missing:
        res.errors.append("Missing required column(s): " + ", ".join(missing) + ".")
        return res
    bad = res.df.index[~res.df["label"].isin(tx.TEXT_LABELS)]
    if len(bad):
        res.errors.append(f"Unknown label on {_lines(bad)}.")
    blank = res.df.index[(res.df["explanation"] == "") | (res.df["group_id"] == "")]
    if len(blank):
        res.errors.append(f"Blank explanation or group_id on {_lines(blank)}.")
    if res.df["example_id"].duplicated().any():
        res.errors.append("example_id values must be unique.")
    return res
