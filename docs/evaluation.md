# Evaluation: written-explanation classifier

> **Synthetic evaluation.** Every explanation was written by the project author from
> templates. There is no real student data and no expert annotation. These results describe
> how well a bag-of-words model separates *these* author-written phrasings. They do **not**
> establish accuracy on real student writing or any effect on teaching or learning.

The machine-readable results are in `artifacts/metrics.json`, the auto-generated tables in
`artifacts/evaluation_report.md`, and the per-example predictions in
`artifacts/test_predictions.csv`. To reproduce them, run `python -m reteach.cli evaluate`
(seed 42).

## 1. Task and classes

The task is to map a short explanation to one of five categories: `correct_reasoning`,
`add_across`, `unscaled_numerator`, `additive_equivalence` and `larger_denominator_bigger`.
Blank, very short (fewer than 3 words) and fully out-of-vocabulary texts are handled by rules
*before* the model runs and are not part of this evaluation.

## 2. Data and split design

| | Dev (train + validation) | Test |
|---|---|---|
| Examples | 130 | 70 |
| Template families (`group_id`) | 65 (13 per class) | 35 (7 per class) |
| Question contexts | C-ADD-1..4, C-EQ-1..3, C-CMP-1..3 | C-ADD-5..6, C-EQ-4..5, C-CMP-4..5 (held out) |

- **Families, not sentences, are the unit of independence.** Each family is one line of
  reasoning written as two paraphrases. A family's paraphrases always land in the same split,
  so the test set never contains a reworded copy of a training sentence. The test asserts
  zero family overlap.
- **Held-out question contexts.** Test explanations are filled with numbers from contexts
  never used in development, such as `1/3 + 1/5` and `2/5 → ?/15`.
- **No double counting.** `explanations.csv` stores each paraphrase once with a dev context
  and once with a test context (400 rows). The split keeps only the row that matches the
  family's split, so each paraphrase is used at most once (200 rows in total). Number
  substitution is not counted as new evidence.
- **Stratified by class.** Per class, one third of the families (7 of 20) go to test, chosen
  with seed 42. Every class has at least 4 development families, which the 4-fold grouped
  cross-validation needs.

## 3. Model and selection (development data only)

- `Pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(class_weight="balanced"))`.
  Because the vectoriser sits inside the pipeline, it is re-fitted on each training fold.
- `C` was chosen by `StratifiedGroupKFold(4)` macro-F1 on dev from {0.3, 1, 3, 10}. The
  results were 0.575, 0.568, 0.545 and 0.533, so **C = 0.3**. Out-of-fold dev accuracy was
  58.5% and macro-F1 0.575.
- The final model is fitted on all 130 dev examples. The same model is used by the app.

## 4. Test results

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression | **64.3%** (45 / 70) | **0.616** |
| `DummyClassifier(most_frequent)` (predicts `add_across`) | 20.0% | 0.067 |

| Class | Precision | Recall | F1 | Support | Test families |
|---|---|---|---|---|---|
| correct_reasoning | 0.43 | 0.43 | 0.43 | 14 | 7 |
| add_across | 0.61 | 0.79 | 0.69 | 14 | 7 |
| unscaled_numerator | 0.60 | 0.21 | 0.32 | 14 | 7 |
| additive_equivalence | 0.67 | 0.86 | 0.75 | 14 | 7 |
| larger_denominator_bigger | 0.87 | 0.93 | 0.90 | 14 | 7 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | correct | add_across | unscaled | additive | larger_den |
|---|---|---|---|---|---|
| correct_reasoning | 6 | 1 | 1 | 4 | 2 |
| add_across | 2 | 11 | 1 | 0 | 0 |
| unscaled_numerator | 6 | 3 | 3 | 2 | 0 |
| additive_equivalence | 0 | 2 | 0 | 12 | 0 |
| larger_denominator_bigger | 0 | 1 | 0 | 0 | 13 |

With only 7 families per class, the per-class numbers are noisy. Moving one family changes a
class's recall by about 0.14.

## 5. Review threshold and selective prediction

The policy was fixed **before** any results were seen: use the *lowest* score threshold at
which the **validation** (out-of-fold dev) accuracy on auto-accepted predictions is at least
90%. If none qualifies, route everything to review.

Validation trade-off (score = max predicted probability, uncalibrated):

| Threshold | Coverage | Accuracy on accepted |
|---|---|---|
| 0.20 | 100% (130) | 58.5% |
| 0.25 | 23.8% (31) | 71.0% |
| 0.30 | 0.8% (1) | 0% |
| ≥ 0.35 | 0% | — |

**No threshold met the policy, so the default routes 100% of predictions to teacher
review.** On test this gives 0% coverage, and the routed predictions are 64.3% accurate.
Test scores range only from 0.21 to 0.30, close to the 0.20 a uniform guess over 5 classes
would give. Strong regularisation (C = 0.3) and little training data make the probabilities
flat, so the score carries little ranking signal. I kept the policy instead of lowering it
after seeing the outcome. The app shows the validation trade-off next to the threshold
control, so a teacher can choose a lower bar knowingly.

Calibration (for example, reliability curves or Brier score) was **not** evaluated, so the
score is treated as a ranking signal, not a probability.

## 6. Error analysis

25 of the 70 test examples are misclassified. The main patterns:

1. **`unscaled_numerator` → `correct_reasoning` (6 errors).** For example, "Common
   denominator is 15, then 1 plus 1 is 2, so 2/15" and "I rewrote 1/3 as 1/15 and 1/5 as
   1/15, then added." The misconception differs from correct reasoning only in *whether the
   numerators were scaled*. Both share the vocabulary "common denominator", "rewrote" and
   "added". The decisive evidence is arithmetic (1/3 ≠ 1/15), and a bag of words can't check
   arithmetic. The default tokenizer also drops single digits, so the numbers are invisible.
2. **`correct_reasoning` → `additive_equivalence` (4 errors).** For example, "The ratio of top
   to bottom has to stay the same, so both get scaled by 3" and "Keeping the same proportion
   means multiplying both parts by 3." The phrases "top and bottom", "both" and "same" are
   strongly associated with the additive misconception in training. The model can't reliably
   tell *multiply both* from *add to both*, because the correct-reasoning families that
   mention "multiply" are few.
3. **`unscaled_numerator` → `add_across` (3 errors).** For example, "The bottoms get
   multiplied together and the tops get added." "Tops get added" is the hallmark phrase of
   add-across. The distinguishing part ("bottoms multiplied") is less common in training.
4. **Vague explanations** such as "the answer is 3 over 9" and "I compared them like normal
   numbers" carry almost no class-specific words, so the prediction is close to a guess.
5. **Novel strategies.** "I used a number line split into 14 parts…" was predicted as
   `larger_denominator_bigger`, because "number line" and "parts" appear mostly in comparison
   explanations in training.

**What would help** (not done, to avoid tuning against this test set): features that parse the
fractions in the text and check them against the question (for example, "rewrote 1/3 as
1/15" is not equivalent), keeping digits in the tokenizer, many more independent families, and
above all real, consented student explanations labelled by trained annotators with an
agreement measure.

## 7. Iteration history (reported for transparency)

1. **Run 1:** 12 families per class (60 in total). Test accuracy was 42.5% and macro-F1 0.436
   (baseline 0.20 / 0.067). Dev cross-validation macro-F1 was 0.468. Trying character n-grams
   and word+char features on dev cross-validation did not help (best 0.465).
2. **Change:** I diagnosed too few independent families (8 per class in dev) from the
   *dev cross-validation* results and wrote 8 new families per class. Dev cross-validation
   macro-F1 rose to 0.575.
3. **Run 2 (reported above):** 64.3% / 0.616.

The test split was therefore viewed twice: once before the data was expanded and once after.
The split procedure, seed, model family and threshold policy did not change between runs, and
no choice was made by comparing test scores. Even so, a fully clean estimate would need a
fresh test set. The difference between runs mostly reflects the larger training set.

## 8. What this evaluation does not show

- That the model works on real students' writing, which is messier, shorter, more varied, and
  includes non-native English and spelling errors.
- That the categories match how expert teachers would label explanations (no inter-rater
  study was done).
- Anything about fairness across student groups (there are no demographic data, and it would
  need real data).
- Any effect on teaching decisions or learning outcomes.
