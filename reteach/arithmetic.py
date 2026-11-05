"""Programmatic arithmetic checks using Python's ``fractions`` module.

Each question can declare an ``operation`` and two operands. From those we
compute the correct answer *and* the answer each misconception would produce,
then check the teacher's answer-key mapping against them.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from . import taxonomy as tx

OPERATIONS = ("add", "equivalent", "compare_larger")

_FRACTION_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(\d+)\s*$")
_INT_RE = re.compile(r"^\s*(-?\d+)\s*$")


def parse_fraction(text: object) -> tuple[int, int] | None:
    """Parse "3/4" -> (3, 4) WITHOUT reducing. Returns None if not a fraction.

    We keep the raw numerator/denominator because "2/6" and "1/3" are the same
    value but tell us different things about how a student worked.
    """
    if text is None:
        return None
    s = str(text)
    m = _FRACTION_RE.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return None
        return num, den
    m = _INT_RE.match(s)
    if m:
        return int(m.group(1)), 1
    return None


def to_value(text: object) -> Fraction | None:
    parsed = parse_fraction(text)
    if parsed is None:
        return None
    return Fraction(parsed[0], parsed[1])


def correct_answer(operation: str, a: str, b: str) -> Fraction:
    """The mathematically correct answer for a question."""
    fa = to_value(a)
    fb = to_value(b)
    if fa is None or fb is None:
        raise ValueError(f"Operands must be fractions or integers: {a!r}, {b!r}")
    if operation == "add":
        return fa + fb
    if operation == "equivalent":
        # a = the fraction, b = the target denominator (integer)
        target_den = int(fb)
        value = fa * target_den
        if value.denominator != 1:
            raise ValueError(f"{a} has no equivalent fraction with denominator {target_den}")
        return Fraction(int(value), target_den)
    if operation == "compare_larger":
        if fa == fb:
            raise ValueError("compare_larger needs two different values")
        return max(fa, fb)
    raise ValueError(f"Unknown operation {operation!r}")


def misconception_answers(operation: str, a: str, b: str) -> dict[str, set[Fraction]]:
    """Values that each misconception would produce for this question.

    Returns {misconception_label: {possible values}}. A misconception that does
    not apply to the operation is simply absent.
    """
    pa, pb = parse_fraction(a), parse_fraction(b)
    if pa is None or pb is None:
        raise ValueError(f"Operands must be fractions or integers: {a!r}, {b!r}")
    (na, da), (nb, db) = pa, pb
    out: dict[str, set[Fraction]] = {}
    if operation == "add":
        out[tx.ADD_ACROSS] = {Fraction(na + nb, da + db)}
        lcd = da * db // math.gcd(da, db)
        # Common denominator found (LCD or product) but numerators left alone.
        out[tx.UNSCALED_NUMERATOR] = {Fraction(na + nb, lcd), Fraction(na + nb, da * db)}
    elif operation == "equivalent":
        target = nb  # b is an integer target denominator, e.g. "8"
        if target > da:
            out[tx.ADDITIVE_EQUIVALENCE] = {Fraction(na + (target - da), target)}
        out[tx.UNSCALED_NUMERATOR] = {Fraction(na, target)}
    elif operation == "compare_larger":
        # Only well-defined for equal numerators: picks the larger denominator.
        if na == nb and da != db:
            out[tx.LARGER_DENOMINATOR] = {Fraction(na, max(da, db))}
    else:
        raise ValueError(f"Unknown operation {operation!r}")
    return out


def check_question(operation: str, a: str, b: str, options: list[dict]) -> list[str]:
    """Check one question's options. Returns a list of human-readable problems.

    ``options`` is a list of dicts with keys option_id, option_text, is_correct,
    misconception_label.
    """
    problems: list[str] = []
    try:
        right = correct_answer(operation, a, b)
        mis = misconception_answers(operation, a, b)
    except ValueError as exc:
        return [str(exc)]

    for opt in options:
        oid = opt["option_id"]
        value = to_value(opt["option_text"])
        label = opt.get("misconception_label") or tx.UNMAPPED
        if opt["is_correct"]:
            if value is None or value != right:
                problems.append(
                    f"option {oid} is marked correct but {opt['option_text']!r} != {right}"
                )
            if label:
                problems.append(f"option {oid} is correct but has a misconception label")
            continue
        if value is not None and value == right:
            problems.append(
                f"option {oid} ({opt['option_text']}) equals the correct answer but is "
                "not marked correct"
            )
        if not label and value is not None:
            for mis_label, values in mis.items():
                if value in values:
                    problems.append(
                        f"option {oid} ({opt['option_text']}) matches the {mis_label!r} "
                        "pattern but has no misconception label"
                    )
        if label:
            if label not in mis:
                problems.append(
                    f"option {oid}: misconception {label!r} does not apply to a "
                    f"{operation!r} question"
                )
            elif value is None or value not in mis[label]:
                expected = ", ".join(str(v) for v in sorted(mis[label]))
                problems.append(
                    f"option {oid}: {opt['option_text']!r} is not what {label!r} "
                    f"produces (expected {expected})"
                )
    return problems
