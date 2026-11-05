# Resume bullets (draft for review; not yet added to the resume)

Each bullet describes only implemented and tested features. The numbers come from
`artifacts/metrics.json` (seed 42) and the test suite.

**Reteach — Student Misconception Analytics** (Python, pandas, scikit-learn, Streamlit,
Plotly, pytest, GitHub Actions) · personal project · synthetic data

- Built a Streamlit dashboard that validates teacher-uploaded quiz CSVs, verifies answer keys
  with Python's `fractions` module, and groups students by misconceptions recurring on at
  least 2 questions, with explicit evidence counts and denominators (demo: 60 synthetic
  students).
- Trained a TF-IDF + logistic regression classifier for written math explanations in a
  scikit-learn Pipeline with group-aware, held-out-context evaluation. It reached 0.62
  macro-F1 versus 0.07 for a majority-class baseline on a synthetic test set, and routes
  low-confidence predictions to teacher review using a threshold chosen on validation data.
- Wrote 51 pytest tests and a GitHub Actions workflow covering input validation, metric
  denominators, train/test leakage, and app smoke tests, and documented the results,
  failure modes and limitations in a model card.

*Wording notes:* keep "synthetic" in the second bullet. Don't add "improved student
outcomes", "deployed" or "used by teachers"; none of these is true.

---

# Five interview questions, with how to answer them

### 1. "Why did you split by template family instead of randomly?"
**Answer:** Each family is one line of reasoning written as two paraphrases. With a random
split, one paraphrase lands in training and its near-copy in testing, so the test measures
memorisation and the score is inflated. I gave each family a `group_id`, kept groups
together, and also held out the numeric question contexts. The test therefore measures
generalisation to new wording *and* new numbers. A test asserts zero group overlap.
**Concept to know:** data leakage; GroupKFold / StratifiedGroupKFold.

### 2. "Your model gets 64% accuracy. Is that good?"
**Answer:** It's relative. A most-frequent baseline gets 20% accuracy and 0.07 macro-F1, so
the model clearly learned something (0.62 macro-F1). But it was evaluated on 70 synthetic
sentences from 35 families, which is small and clean. One class (`unscaled_numerator`) has
recall of only 0.21. I wouldn't trust it to label student work without review, which is why
the app routes everything to teacher review.
**Concept to know:** baselines; macro- versus micro-averaging; why per-class metrics matter.

### 3. "Why does your app send every prediction to review?"
**Answer:** Before seeing any results, I set a policy: auto-accept only above the lowest
threshold where validation accuracy on accepted predictions reaches 90%. On out-of-fold
validation, the best threshold reached 71% accuracy at 24% coverage, so nothing qualified.
The model's top probabilities sit around 0.21–0.30 because it's strongly regularised and
trained on little data. Lowering the bar after seeing that would just be moving the goalposts.
Teachers can still change the threshold, with the validation trade-off shown next to the
control.
**Concept to know:** selective prediction, coverage versus accuracy, calibration, and why the
threshold must come from validation data and not test data.

### 4. "What's ML here and what isn't?"
**Answer:** The misconception counts and student groups are rules. They come from the
teacher's answer key, where each wrong option is mapped to a misconception, and require at
least 2 distinct supporting questions before a pattern is called recurring. Only the
written-explanation categories come from ML. The UI badges them differently, because calling
a lookup table "AI" would mislead teachers about how certain the results are.
**Concept to know:** interpretability; the difference between a diagnostic rule and a
statistical prediction.

### 5. "What would you do next to make this real?"
**Answer:**
1. Collect real, consented, de-identified explanations.
2. Have trained annotators label them and measure inter-rater agreement (for example,
   Cohen's kappa).
3. Re-evaluate, including a calibration check and error analysis by subgroup.
4. Add features that parse the fractions in the text, since the main errors are arithmetic
   that a bag of words can't see.
5. Run a usability study with teachers before any claim about classroom value.

**Concept to know:** annotation quality, distribution shift, and the gap between offline
metrics and real-world impact.
