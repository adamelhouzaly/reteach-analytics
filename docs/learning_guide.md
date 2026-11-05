# Learning guide: how Reteach works

This guide is for someone still learning Python and ML. It walks through the code in the
order data flows through it, then explains the ML ideas you'll be asked about.

## Part 1: the code, step by step

### 1. `reteach/taxonomy.py`: the vocabulary of the project
This file holds plain constants, such as `ADD_ACROSS = "add_across"`, and a `Category`
**dataclass** for each misconception. A dataclass is a class that mainly stores data. Writing
`@dataclass(frozen=True)` gives you a tidy record whose fields can't be changed by accident.
Nothing in this file is "smart". It's hand-written knowledge, which is exactly why the app
labels it as a teacher-defined rule.

### 2. `reteach/arithmetic.py`: checking the answer key with `fractions`
`Fraction(1, 2) + Fraction(1, 3)` gives exactly `Fraction(5, 6)`, with no floating-point
rounding.
- `parse_fraction("2/6")` returns `(2, 6)` *without* reducing it, because `2/6` versus `1/3`
  tells you how the student worked.
- `correct_answer(op, a, b)` computes the right answer.
- `misconception_answers(op, a, b)` computes what each wrong idea would produce. For example,
  add-across of 1/2 and 1/3 is `(1+1)/(2+3) = 2/5`.
- `check_question(...)` compares the teacher's answer key with both and returns a list of
  problems.

### 3. `reteach/validation.py`: never trust a CSV
- `read_csv` reads every column as text (`dtype=str`) so nothing is silently converted, and
  catches empty or unreadable files.
- Validators return a `ValidationResult` with a cleaned DataFrame, **errors** (stop) and
  **warnings** (fixed for you, but you should know). The app prints these to the teacher, so
  users never see a Python traceback.
- Row numbers in messages are `index + 2`, because CSV line 1 is the header.

### 4. `reteach/analytics.py`: counting with clear denominators
- `score_responses` **merges** (joins) responses with the answer key, so each response row
  knows its option text, whether it's correct, and its misconception label.
- `coverage` returns the counts *and* the denominators: expected = students × questions;
  answered; unanswered (blank); missing (no row).
- `misconception_summary` reports "chosen / opportunities". An *opportunity* is an answered
  question that offered that distractor. Without a denominator, "60 students chose it" means
  nothing.
- `suggest_topics` applies two simple, explainable rules and records the *reason* for each
  suggestion.

### 5. `reteach/grouping.py`: one wrong answer is not a diagnosis
For each student and misconception it counts the **distinct questions** where the student
picked a matching distractor. If the count is `>= min_support` (default 2), the pattern is
*recurring*; 1 is *possible*. Students with the same set of recurring misconceptions form a
group. Look for `groupby` and `agg`. These are the pandas "split, apply, combine" tools.

### 6. `reteach/model.py`: the classifier
```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(C=C, class_weight="balanced")),
])
```
- `split_dataset` creates the train and test sets (explained in Part 2).
- `select_C` tries a few regularisation strengths using cross-validation on training data only.
- `choose_threshold` picks the review threshold from validation predictions.
- `classify` first applies rules (blank, too short, unknown words), then the model, then the
  threshold.

### 7. `reteach/evaluate.py`: measuring honestly
It trains, predicts on the untouched test set, computes metrics, compares against a
`DummyClassifier`, and writes `artifacts/metrics.json` plus a readable report.

### 8. `app.py`, `views/` and `ui/`: presentation only
Streamlit reruns the whole script top to bottom on every click.
- `app.py` sets up the page and the top navigation (`st.navigation`). Each page is a file
  in `views/`.
- `ui/state.py` holds `st.session_state`, which remembers things between reruns, such as
  the current dataset, the confirmed labels and the threshold setting. Widgets copy their
  value into plain state keys through `on_change`, so a setting survives page changes.
  `@st.cache_resource` keeps the trained model in memory.
- `ui/theme.py` holds the colour tokens and a small amount of CSS. That CSS targets only
  classes we create with `st.container(key="card-...")`.
- `ui/components.py` has reusable pieces such as the page header, cards, tags and empty
  states.
- `app.py` wraps each page in a `try/except`, so teachers see a friendly message instead
  of a traceback.

## Part 2: ML concepts in plain language

**TF-IDF.** TF-IDF turns text into numbers. Each word (and two-word phrase) becomes a column.
The value is high when a word appears in this explanation (term frequency) but is rare across
all explanations (inverse document frequency). So "denominators" counts more than "the".
Word *order* and arithmetic are lost. This is a "bag of words".

**Logistic regression.** This learns one weight per word per class. To classify, it adds up
the weights of the words present and turns the totals into probabilities that sum to 1 (a
softmax). `C` controls regularisation: a *smaller* C keeps the weights small, which reduces
overfitting but makes the probabilities flatter, as you can see in this project.

**Pipeline.** Bundling the vectoriser and classifier means that `fit` learns the vocabulary
from *training* text only. If you fitted TF-IDF on all the data first, test words would leak
into training. That is a subtle form of **data leakage**.

**Group-aware splitting.** Two paraphrases of the same idea are nearly the same example. If
one is in training and the other is in testing, the test only checks memory. Giving each
family a `group_id` and keeping groups together tests the realistic case of *new* reasoning
wording. `StratifiedGroupKFold` does this for cross-validation while keeping the classes
balanced in each fold.

**Train / validation / test.** You *learn* on training data, *make choices* (C, threshold) on
validation data, and *report* on test data. If you make choices by looking at test scores,
the test score becomes optimistic. This project reports that the test set was viewed twice
(see `docs/evaluation.md` §7).

**Baseline.** A `DummyClassifier` that always predicts the most common class. If a model
can't beat this, it has learned nothing useful. Here the baseline gets 20% accuracy (5
balanced classes).

**Precision, recall, F1.** For one class: *precision* answers "when the model says
add-across, how often is it right?", and *recall* answers "of all real add-across
explanations, how many did it find?". *F1* is their harmonic mean. **Macro F1** averages F1
across classes equally, so a model can't hide a terrible class behind good ones. *Support* is
how many true examples a class has.

**Confusion matrix.** Rows are the true class and columns are the predicted class. The
diagonal shows correct predictions, and off-diagonal cells show *which* classes get
confused. That's where error analysis starts.

**Selective prediction and thresholds.** If the model's top probability is below a threshold,
don't trust it automatically; send it to a human. *Coverage* is the share auto-accepted.
Raising the threshold usually lowers coverage and raises accuracy on what's accepted.

**Calibration.** A model is calibrated if, among predictions scored 0.8, about 80% are
right. Logistic regression *often* is, but not always, and this project didn't measure it.
So the score is treated as a ranking signal only.

**Synthetic data.** It's useful for building and testing a pipeline, but its numbers say
little about the real world. The model is mainly learning the author's writing style.
