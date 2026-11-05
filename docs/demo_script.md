# 90-second recruiter walkthrough

Start the app with `streamlit run app.py`. The synthetic demo loads automatically. If you
clicked around earlier, press **Reset demo** in the sidebar first.

---

**0:00 — Framing (Overview tab)**
> "Reteach helps a teacher decide what to reteach after a fraction quiz. Everything here is
> synthetic: 60 fictional students and explanations I wrote myself. There are two kinds of
> output, and the app keeps them apart: blue badges are teacher-defined rules from the
> answer key, and orange badges are machine-learning suggestions."

**0:15 — The decision**
> "Coverage is 96% and accuracy is 59%. Blank and missing answers are excluded from
> accuracy and shown separately. 'Adding unlike denominators' is suggested for review, and
> the app says why: accuracy is 55%, and two misconceptions recur for 27% of students. The
> suggested activity is guidance I authored, not AI-generated."

**0:30 — Evidence (Misconceptions tab)**
> "'Adds numerators and denominators' was chosen 60 times out of 229 opportunities. Here are
> the supporting questions, the student IDs, and students with only one supporting question,
> which I label as insufficient evidence instead of a diagnosis."

**0:42 — Groups (Student groups tab)**
> "Students are grouped by misconceptions that recur on at least two distinct questions."
> *Drag the sidebar slider to 1, then back to 2.*
> "With a threshold of 1, groups balloon. That's why the default is 2 and teachers can adjust it."

**0:55 — ML with guardrails (Written-answer review tab)**
> "A TF-IDF plus logistic-regression pipeline suggests a category for each written
> explanation. Blank, very short and unrecognized texts are never forced into a category. I
> chose the auto-accept threshold on validation data with a 90% accuracy policy. The model
> didn't meet it, so every suggestion goes to teacher review."
> *Pick an explanation, choose a category, click Save correction.*
> "Corrections are saved with provenance in a separate file and never retrain the model
> automatically."

**1:10 — Honest evaluation (Model evaluation tab)**
> "On a held-out test set, with template families and number contexts the model never saw,
> it gets 64% accuracy and 0.62 macro-F1, against 20% and 0.07 for a majority baseline. The
> confusion matrix shows the main weakness: 'unscaled numerator' looks like correct
> reasoning to a bag-of-words model, because the difference is arithmetic, not vocabulary."

**1:25 — Close (Data upload tab)**
> "Teachers can upload their own CSVs. Validation explains exactly what's wrong, down to line
> numbers, and the answer key is checked with Python's fractions module."

---

**If asked "does it work in classrooms?":** "I don't know. It hasn't been tested with real
students or teachers. The next step would be real, consented data with expert labels."
