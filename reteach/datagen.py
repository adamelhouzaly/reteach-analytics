"""Generate all SYNTHETIC data shipped with the project.

Run:  python -m reteach.cli generate-data

Outputs (all deterministic for a fixed seed):
  data/demo/questions.csv         10-question fraction quiz (answer key)
  data/demo/responses.csv         60 fictional students
  data/labeled/contexts.csv       numeric contexts for explanation templates
  data/labeled/explanations.csv   labeled explanations for the classifier
  data/templates/*.csv            empty upload templates
  data/invalid/*.csv              deliberately broken inputs for tests/demo
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from . import RANDOM_SEED
from . import explanation_bank as bank
from . import taxonomy as tx
from .arithmetic import parse_fraction

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SYNTHETIC_NOTE = "synthetic"

# --- Quiz ------------------------------------------------------------------
TOPIC_EQ = "Equivalent fractions"
TOPIC_CMP = "Comparing fractions"
TOPIC_ADD = "Adding unlike denominators"
EXPLAIN = " Explain how you got your answer."

A, U, AE, LD, N = (tx.ADD_ACROSS, tx.UNSCALED_NUMERATOR, tx.ADDITIVE_EQUIVALENCE,
                   tx.LARGER_DENOMINATOR, tx.UNMAPPED)
EQUAL, NEI = "They are equal", "Not enough information"

# (qid, topic, operation, a, b, ask_explanation, [(option_text, is_correct, label)])
QUIZ = [
    ("Q1", TOPIC_EQ, "equivalent", "1/2", "8", False,
     [("4/8", True, N), ("7/8", False, AE), ("1/8", False, U), ("2/8", False, N)]),
    ("Q2", TOPIC_EQ, "equivalent", "2/3", "9", True,
     [("8/9", False, AE), ("6/9", True, N), ("2/9", False, U), ("3/9", False, N)]),
    ("Q3", TOPIC_EQ, "equivalent", "3/4", "12", False,
     [("3/12", False, U), ("11/12", False, AE), ("6/12", False, N), ("9/12", True, N)]),
    ("Q4", TOPIC_CMP, "compare_larger", "1/3", "1/5", False,
     [("1/3", True, N), ("1/5", False, LD), (EQUAL, False, N), (NEI, False, N)]),
    ("Q5", TOPIC_CMP, "compare_larger", "2/7", "2/9", True,
     [("2/9", False, LD), ("2/7", True, N), (EQUAL, False, N), (NEI, False, N)]),
    ("Q6", TOPIC_CMP, "compare_larger", "3/4", "3/8", False,
     [(EQUAL, False, N), ("3/8", False, LD), ("3/4", True, N), (NEI, False, N)]),
    ("Q7", TOPIC_ADD, "add", "1/2", "1/3", False,
     [("2/5", False, A), ("5/6", True, N), ("2/6", False, U), ("1/6", False, N)]),
    ("Q8", TOPIC_ADD, "add", "1/4", "2/3", True,
     [("11/12", True, N), ("3/7", False, A), ("3/12", False, U), ("2/12", False, N)]),
    ("Q9", TOPIC_ADD, "add", "3/5", "1/10", False,
     [("4/10", False, U), ("7/10", True, N), ("4/15", False, A), ("7/15", False, N)]),
    ("Q10", TOPIC_ADD, "add", "2/3", "1/6", True,
     [("3/9", False, A), ("3/6", False, U), ("5/6", True, N), ("4/6", False, N)]),
]


def _prompt(operation: str, a: str, b: str) -> str:
    if operation == "equivalent":
        return f"Which fraction is equivalent to {a} and has denominator {b}?"
    if operation == "compare_larger":
        return f"Which fraction is larger: {a} or {b}?"
    return f"What is {a} + {b}?"


def build_questions() -> pd.DataFrame:
    rows = []
    for qid, topic, op, a, b, explain, options in QUIZ:
        prompt = _prompt(op, a, b) + (EXPLAIN if explain else "")
        for letter, (text, correct, label) in zip("ABCD", options):
            rows.append({
                "question_id": qid, "topic": topic, "prompt": prompt, "option_id": letter,
                "option_text": text, "is_correct": "TRUE" if correct else "FALSE",
                "misconception_label": label, "operation": op, "operand_a": a, "operand_b": b,
            })
    return pd.DataFrame(rows)


# --- Template contexts -----------------------------------------------------
# (context_id, operation, a, b, context_split). "test" contexts are reserved
# for held-out evaluation so the model is tested on unseen numbers too.
CONTEXTS = [
    ("C-ADD-1", "add", "1/2", "1/3", "dev"),
    ("C-ADD-2", "add", "1/4", "2/3", "dev"),
    ("C-ADD-3", "add", "2/5", "1/3", "dev"),
    ("C-ADD-4", "add", "3/4", "1/6", "dev"),
    ("C-ADD-5", "add", "1/3", "1/5", "test"),
    ("C-ADD-6", "add", "2/7", "1/2", "test"),
    ("C-EQ-1", "equivalent", "1/2", "8", "dev"),
    ("C-EQ-2", "equivalent", "2/3", "9", "dev"),
    ("C-EQ-3", "equivalent", "3/5", "10", "dev"),
    ("C-EQ-4", "equivalent", "2/5", "15", "test"),
    ("C-EQ-5", "equivalent", "1/3", "12", "test"),
    ("C-CMP-1", "compare_larger", "1/3", "1/5", "dev"),
    ("C-CMP-2", "compare_larger", "2/7", "2/9", "dev"),
    ("C-CMP-3", "compare_larger", "1/4", "1/6", "dev"),
    ("C-CMP-4", "compare_larger", "3/8", "3/5", "test"),
    ("C-CMP-5", "compare_larger", "1/2", "1/9", "test"),
]


def context_values(operation: str, a: str, b: str) -> dict:
    """Numbers used to fill explanation templates for one context."""
    (na, da), (nb, db) = parse_fraction(a), parse_fraction(b)
    v = {"a": a, "b": b, "na": na, "da": da, "nb": nb, "db": db}
    if operation == "add":
        lcd = da * db // math.gcd(da, db)
        fa, fb = lcd // da, lcd // db
        v.update(lcd=lcd, fa=fa, fb=fb, sa=na * fa, sb=nb * fb, prod=da * db,
                 ans=f"{na * fa + nb * fb}/{lcd}", sum_n=na + nb, sum_d=da + db,
                 across=f"{na + nb}/{da + db}", unscaled=f"{na + nb}/{lcd}")
    elif operation == "equivalent":
        target = nb
        factor = target // da
        diff = target - da
        v.update(target=target, factor=factor, new_n=na * factor, ans=f"{na * factor}/{target}",
                 diff=diff, add_n=na + diff, additive=f"{na + diff}/{target}", gap=da - na)
    else:
        big, small = (a, b) if Fraction(a) > Fraction(b) else (b, a)
        v.update(big=big, small=small, n=na, dfew=min(da, db), dmany=max(da, db))
    return v


def build_labeled(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each paraphrase is instantiated once in a dev context and once in a test
    context. The split (reteach.model.split_dataset) keeps only the row that
    matches its family's split, so a paraphrase is used at most once."""
    rng = np.random.default_rng(seed)
    ctx = pd.DataFrame(CONTEXTS, columns=["context_id", "operation", "operand_a", "operand_b",
                                          "context_split"])
    rows = []
    for label, families in bank.FAMILIES.items():
        for fi, (op, paraphrases) in enumerate(families, start=1):
            group_id = f"{label}-F{fi:02d}"
            for pi, template in enumerate(paraphrases, start=1):
                for split in ("dev", "test"):
                    options = ctx[(ctx["operation"] == op) & (ctx["context_split"] == split)]
                    c = options.iloc[rng.integers(len(options))]
                    text = template.format(**context_values(op, c["operand_a"], c["operand_b"]))
                    rows.append({
                        "example_id": f"{group_id}-P{pi}-{split}",
                        "question_id": c["context_id"],
                        "explanation": text,
                        "label": label,
                        "source": "synthetic_template_author_written",
                        "group_id": group_id,
                    })
    return pd.DataFrame(rows), ctx


# --- Fictional students ----------------------------------------------------
# profile -> (count, {misconception: probability of choosing it when offered},
#             probability correct otherwise)
PROFILES = {
    "secure": (15, {}, 0.92),
    "add_across": (9, {A: 0.75}, 0.80),
    "unscaled": (9, {U: 0.65}, 0.80),
    "additive": (6, {AE: 0.75}, 0.80),
    "larger_denominator": (7, {LD: 0.75}, 0.80),
    "across_and_denominator": (4, {A: 0.7, LD: 0.7}, 0.75),
    "developing": (6, {}, 0.55),
    "guessing": (4, {}, 0.25),
}
P_BLANK_ANSWER = 0.03
P_BLANK_EXPLANATION = 0.10
P_VAGUE_EXPLANATION = 0.08


def build_responses(questions: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    q_rows = {qid: g for qid, g in questions.groupby("question_id", sort=False)}
    ctx_by_q = {q[0]: context_values(q[2], q[3], q[4]) for q in QUIZ}
    op_by_q = {q[0]: q[2] for q in QUIZ}
    explain_q = {q[0] for q in QUIZ if q[5]}

    profiles = [p for p, (n, _, _) in PROFILES.items() for _ in range(n)]
    rng.shuffle(profiles)
    rows = []
    for i, profile in enumerate(profiles, start=1):
        sid = f"S{i:03d}"
        _, tendencies, p_correct = PROFILES[profile]
        # Two students were absent for the last two questions (missing rows).
        skip = {"Q9", "Q10"} if i in (17, 44) else set()
        for qid, opts in q_rows.items():
            if qid in skip:
                continue
            if rng.random() < P_BLANK_ANSWER:
                rows.append({"student_id": sid, "question_id": qid, "selected_option": "",
                             "explanation": ""})
                continue
            chosen = None
            for label, p in tendencies.items():
                mapped = opts[opts["misconception_label"] == label]
                if len(mapped) and rng.random() < p:
                    chosen = mapped.iloc[0]
                    break
            if chosen is None:
                if rng.random() < p_correct:
                    chosen = opts[opts["is_correct"] == "TRUE"].iloc[0]
                else:
                    wrong = opts[opts["is_correct"] == "FALSE"]
                    chosen = wrong.iloc[rng.integers(len(wrong))]
            explanation = ""
            if qid in explain_q:
                explanation = _explain(rng, chosen, op_by_q[qid], ctx_by_q[qid])
            rows.append({"student_id": sid, "question_id": qid,
                         "selected_option": chosen["option_id"], "explanation": explanation})
    return pd.DataFrame(rows)


def _explain(rng, option, operation: str, values: dict) -> str:
    r = rng.random()
    if r < P_BLANK_EXPLANATION:
        return ""
    label = tx.CORRECT if option["is_correct"] == "TRUE" else option["misconception_label"]
    pool = bank.DEMO_EXPLANATIONS.get((label, operation))
    if r < P_BLANK_EXPLANATION + P_VAGUE_EXPLANATION or not pool:
        return bank.DEMO_VAGUE[rng.integers(len(bank.DEMO_VAGUE))]
    return pool[rng.integers(len(pool))].format(**values)


# --- Templates & invalid examples -----------------------------------------
def write_templates(questions: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    questions[questions["question_id"] == "Q7"].to_csv(out / "questions_template.csv", index=False)
    pd.DataFrame([
        {"student_id": "S001", "question_id": "Q7", "selected_option": "B",
         "explanation": "I used 6 as the common denominator: 3/6 + 2/6 = 5/6"},
        {"student_id": "S002", "question_id": "Q7", "selected_option": "", "explanation": ""},
    ]).to_csv(out / "responses_template.csv", index=False)


def write_invalid(questions: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    q = questions.copy()
    two_correct = q.copy()
    two_correct.loc[(two_correct["question_id"] == "Q7") & (two_correct["option_id"] == "A"),
                    "is_correct"] = "TRUE"
    two_correct.to_csv(out / "questions_two_correct.csv", index=False)

    wrong_math = q.copy()
    wrong_math.loc[(wrong_math["question_id"] == "Q8") & (wrong_math["option_id"] == "A"),
                   "option_text"] = "10/12"
    wrong_math.to_csv(out / "questions_wrong_answer_key.csv", index=False)

    q.drop(columns=["is_correct"]).to_csv(out / "questions_missing_column.csv", index=False)
    (out / "empty.csv").write_text("")

    pd.DataFrame([
        {"student_id": "S001", "question_id": "Q7", "selected_option": "B", "explanation": "ok"},
        {"student_id": "S001", "question_id": "Q7", "selected_option": "A",
         "explanation": "duplicate row"},
        {"student_id": "S002", "question_id": "Q99", "selected_option": "A",
         "explanation": "unknown question"},
        {"student_id": "S003", "question_id": "Q7", "selected_option": "Z",
         "explanation": "invalid choice"},
        {"student_id": "", "question_id": "Q7", "selected_option": "B",
         "explanation": "missing student id"},
        {"student_id": "S004", "question_id": "Q8", "selected_option": "", "explanation": ""},
    ]).to_csv(out / "responses_messy.csv", index=False)


def generate_all(seed: int = RANDOM_SEED, root: Path = DATA) -> dict[str, int]:
    questions = build_questions()
    responses = build_responses(questions, seed)
    labeled, contexts = build_labeled(seed)
    (root / "demo").mkdir(parents=True, exist_ok=True)
    (root / "labeled").mkdir(parents=True, exist_ok=True)
    questions.to_csv(root / "demo" / "questions.csv", index=False)
    responses.to_csv(root / "demo" / "responses.csv", index=False)
    labeled.to_csv(root / "labeled" / "explanations.csv", index=False)
    contexts.to_csv(root / "labeled" / "contexts.csv", index=False)
    write_templates(questions, root / "templates")
    write_invalid(questions, root / "invalid")
    return {"questions": questions["question_id"].nunique(),
            "students": responses["student_id"].nunique(),
            "responses": len(responses), "labeled_rows": len(labeled),
            "families": labeled["group_id"].nunique()}
