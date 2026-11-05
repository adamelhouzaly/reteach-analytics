"""The misconception taxonomy and teacher-authored review guidance.

Everything in this file is *hand-written, deterministic content*. Nothing
here is produced by machine learning. The app labels it as
"teacher-defined" so it is never confused with model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Category labels -------------------------------------------------------
CORRECT = "correct_reasoning"
ADD_ACROSS = "add_across"
UNSCALED_NUMERATOR = "unscaled_numerator"
ADDITIVE_EQUIVALENCE = "additive_equivalence"
LARGER_DENOMINATOR = "larger_denominator_bigger"

MISCONCEPTIONS = [ADD_ACROSS, UNSCALED_NUMERATOR, ADDITIVE_EQUIVALENCE, LARGER_DENOMINATOR]
# Labels the text classifier can predict.
TEXT_LABELS = [CORRECT, *MISCONCEPTIONS]

# --- Handling states (NOT model classes) -----------------------------------
# These are assigned by simple rules *before or after* the model runs, so the
# system never forces a weak or empty answer into a misconception.
STATE_BLANK = "blank"  # no explanation written
STATE_INSUFFICIENT = "insufficient_text"  # too short to judge (e.g. "idk")
STATE_UNRECOGNIZED = "unrecognized"  # no words the model has ever seen
STATE_NEEDS_REVIEW = "needs_review"  # model score below the review threshold
STATE_AUTO = "auto_accepted"  # model score at/above threshold
STATE_CORRECTED = "teacher_corrected"  # a teacher has set the label

# Placeholder for multiple-choice distractors that are wrong but not tied to
# a specific misconception. Picking one is "unrecognized error", not a diagnosis.
UNMAPPED = ""


@dataclass(frozen=True)
class Category:
    label: str
    name: str
    short: str
    description: str
    example: str
    review_activity: str


CATEGORIES: dict[str, Category] = {
    CORRECT: Category(
        label=CORRECT,
        name="Correct reasoning",
        short="Correct",
        description="Rewrites both fractions over a common denominator, scaling "
        "numerator and denominator by the same factor, then adds numerators.",
        example="1/2 + 1/3 = 3/6 + 2/6 = 5/6",
        review_activity="No reteach needed. Offer an extension: ask the student to "
        "explain why the least common denominator is convenient but not required.",
    ),
    ADD_ACROSS: Category(
        label=ADD_ACROSS,
        name="Adds numerators and denominators",
        short="Add across",
        description="Treats a fraction as two separate whole numbers and adds "
        "top-to-top and bottom-to-bottom.",
        example="1/2 + 1/3 = 2/5  (correct: 5/6). Note 2/5 < 1/2, so the 'sum' is "
        "smaller than one of the addends.",
        review_activity="Estimate first: is 1/2 + 1/3 more or less than 1/2? Use "
        "fraction strips or a number line to show that 2/5 is smaller than 1/2, so "
        "it cannot be the sum. Then model adding same-sized pieces.",
    ),
    UNSCALED_NUMERATOR: Category(
        label=UNSCALED_NUMERATOR,
        name="Changes denominator without scaling numerator",
        short="Unscaled numerator",
        description="Finds or uses a new denominator but keeps the original "
        "numerator(s), so the rewritten fraction is no longer equivalent.",
        example="1/2 + 1/3 → 1/6 + 1/6 = 2/6  (correct: 3/6 + 2/6 = 5/6); "
        "or 2/3 = 2/9  (correct: 6/9).",
        review_activity="Area-model check: shade 1/2 of a rectangle, then cut it "
        "into sixths. Count the shaded sixths (3, not 1). Practise 'whatever you do "
        "to the bottom, do to the top' with the multiplier written explicitly.",
    ),
    ADDITIVE_EQUIVALENCE: Category(
        label=ADDITIVE_EQUIVALENCE,
        name="Adds (not multiplies) to make equivalent fractions",
        short="Additive equivalence",
        description="Adds the same number to numerator and denominator, "
        "believing this keeps the fraction's value.",
        example="1/2 = ?/8 → adds 6 to both → 7/8  (correct: multiply by 4 → 4/8).",
        review_activity="Test the rule on 1/2 → 2/3 by adding 1: draw both and show "
        "they are different sizes. Contrast with 1/2 → 2/4 (multiply by 2). Use a "
        "multiplication table to find equivalent fractions.",
    ),
    LARGER_DENOMINATOR: Category(
        label=LARGER_DENOMINATOR,
        name="Larger denominator means larger fraction",
        short="Bigger denominator",
        description="Applies whole-number ordering to denominators: thinks 1/5 > 1/3 "
        "because 5 > 3.",
        example="Which is larger, 1/3 or 1/5? → chooses 1/5  (correct: 1/3, because "
        "thirds are bigger pieces than fifths).",
        review_activity="Fair-share task: one pizza shared by 3 people vs. 5 people — "
        "who gets more? Place unit fractions on a number line from 0 to 1.",
    ),
}

STATE_DESCRIPTIONS = {
    STATE_BLANK: "No written explanation was provided.",
    STATE_INSUFFICIENT: "Fewer than 3 words — too little to categorise.",
    STATE_UNRECOGNIZED: "Contains no words the model saw in training.",
    STATE_NEEDS_REVIEW: "Model score below the review threshold — teacher should check.",
    STATE_AUTO: "Model score at or above the review threshold.",
    STATE_CORRECTED: "Label set by a teacher.",
}


def display_name(label: str) -> str:
    """Human-readable name for a category label or handling state."""
    if label in CATEGORIES:
        return CATEGORIES[label].name
    if not label:
        return "Unmapped wrong answer"
    return label.replace("_", " ").capitalize()


def short_name(label: str) -> str:
    if label in CATEGORIES:
        return CATEGORIES[label].short
    return display_name(label)
