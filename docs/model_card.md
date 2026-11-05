# Model card: Reteach explanation classifier

## Overview

| | |
|---|---|
| Model | scikit-learn `Pipeline`: `TfidfVectorizer` (word 1–2-grams, sublinear TF) → `LogisticRegression` (C = 0.3, balanced class weights) |
| Version | 0.1.0, seed 42 |
| Input | A short written explanation (English) of a fraction quiz answer |
| Output | One of 5 categories and a max-probability score (uncalibrated) |
| Training data | 130 author-written synthetic explanations (65 template families) |
| Artifacts | Trained locally into `models/` (not committed); metrics in `artifacts/metrics.json` |

## Intended use

- **Triage support for a teacher** who reviews student explanations from a fraction-addition
  quiz. The model suggests a category, and the teacher decides.
- **A portfolio demonstration** of an end-to-end, honestly evaluated text classifier.

Predictions are shown with an "ML suggestion" badge and a review status. Under the default
threshold **every prediction is routed to teacher review** (see Evaluation).

## Out of scope

- Diagnosing individual students, grading, placement, or any high-stakes decision.
- Any subject or skill other than adding fractions with unlike denominators and the two
  prerequisites included (equivalent fractions, comparing fractions with equal numerators).
- Non-English text, speech transcripts, or images of handwritten work.
- Automatic retraining from teacher corrections. Corrections are stored separately with
  provenance and require a reviewed workflow.
- Claims about learning outcomes or classroom effectiveness.

## Categories

| Label | Meaning |
|---|---|
| `correct_reasoning` | Common denominator with numerators scaled; multiply both parts for equivalence; fewer pieces means bigger pieces |
| `add_across` | Adds numerators and adds denominators (1/2 + 1/3 = 2/5) |
| `unscaled_numerator` | New denominator, original numerator (1/2 + 1/3 = 2/6; 2/3 = 2/9) |
| `additive_equivalence` | Adds the same number to top and bottom (1/2 = 7/8) |
| `larger_denominator_bigger` | Thinks 1/5 > 1/3 because 5 > 3 |

Rules run before the model: **blank**, **insufficient text** (fewer than 3 words) and
**unrecognized** (no known vocabulary) are never given a category.

## Training data

- **Source:** written by the project author (`reteach/explanation_bank.py`). There are 20
  families per class, each with 2 paraphrases, filled with numbers from 16 question contexts.
  `source = synthetic_template_author_written`.
- **Not** real student work, not crowd-sourced, and not expert-annotated. Labels are correct
  by construction, because the author wrote each template to express one category.
- Known biases: the text is cleaner and more explicit than children's writing, has balanced
  classes, and its vocabulary reflects one author.

## Evaluation (synthetic held-out test)

The test set has 70 examples from 35 unseen template families, on unseen numeric contexts.

| | Accuracy | Macro F1 |
|---|---|---|
| This model | 64.3% | 0.616 |
| Most-frequent baseline | 20.0% | 0.067 |

The weakest class is `unscaled_numerator` (recall 0.21), and the strongest is
`larger_denominator_bigger` (F1 0.90). Validation never met the 90%-accuracy auto-accept policy,
so test coverage at the default threshold is 0%. Calibration was not evaluated. See
[evaluation.md](evaluation.md).

## Failure modes

- **Cannot check arithmetic.** "I rewrote 1/3 as 1/15" sounds like correct reasoning, and the
  model has no way to see it isn't. Single digits are dropped by the tokenizer.
- **Shared vocabulary.** "Same to top and bottom" appears in both correct (multiply) and
  additive (add) reasoning.
- **Vague or novel strategies** (number lines, "like normal numbers") get near-uniform scores.
- **Flat scores.** The maximum probabilities cluster around 0.21–0.30, so the score is a weak
  ranking signal.
- **Distribution shift.** Real student language (spelling, brevity, mixed reasoning) is
  likely to lower performance by an unknown amount.

## Ethical considerations

- Student IDs are pseudonymous, and the app asks users not to upload names.
- A wrong label could bias a teacher's view of a student. Mitigations: the review-first
  default, visible scores, the separation of rules from ML, and grouping that requires
  repeated multiple-choice evidence.
- Before any real use, the model would need real, consented data, expert labels with an
  inter-rater agreement measure, a calibration check, subgroup error analysis, and teacher
  usability testing.
