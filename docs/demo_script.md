# 90-second recruiter walkthrough

Start the app with `streamlit run app.py`. The demo class loads automatically. If you
clicked around earlier, use **Upload data → Reset to demo data** (or refresh the browser).

---

**0:00 — Framing (Overview)**
> "Reteach helps a teacher decide what to reteach after a fraction quiz. The 'Demo data'
> badge means everything here is synthetic: 60 fictional students and explanations I wrote
> myself. Teal tags are rules from the answer key, and indigo tags are machine-learning
> suggestions. The app never mixes the two."

**0:12 — The decision**
> "The four cards answer 'how did the class do?': 59% accuracy and 96% coverage. Blank and
> missing answers aren't counted as wrong. 'What to reteach next' names *Adding unlike
> denominators*: it's below the 70% target, and two misconceptions recur for 16 students
> each. Next to that is a short activity I authored, not AI-generated."

**0:28 — Evidence**
> *Click **See misconception evidence**.*
> "The selected misconception opens first. 26 of 60 students chose 'adds the tops and the
> bottoms' at least once, but only 16 did it on two or more questions. One wrong answer is
> evidence, not a diagnosis."
> *Expand **Students**.*

**0:42 — Groups (Student groups)**
> *Switch the threshold to **1 question**, then back to **2**.*
> "With a threshold of one, the groups balloon. Students without a recurring pattern stay
> separate from students who got everything right, because those aren't the same thing."

**0:55 — ML with guardrails (Review answers)**
> "A TF-IDF plus logistic-regression model suggests a category for each written
> explanation. The score is a ranking score, not a probability. I set a 90%-accuracy rule on
> validation data before seeing results. The model didn't meet it, so every suggestion goes
> to teacher review."
> *Pick a category and click **Save category**; the next answer appears.*
> "Confirmed labels go to a separate file with provenance and never retrain the model."

**1:10 — Honest evaluation (Model evaluation)**
> "On held-out test data, with template families and numbers the model never saw, it scores
> 64% accuracy and 0.62 macro-F1, against 20% and 0.07 for a majority baseline. The
> confusion matrix shows the weakness: 'changes the bottom but not the top' reads like
> correct reasoning to a bag-of-words model, because the difference is arithmetic, not
> wording."

**1:25 — Close (Upload data)**
> "Teachers can upload their own CSVs. Validation explains exactly what's wrong, down to line
> numbers, and the answer key is checked with Python's fractions module."

---

**If asked "does it work in classrooms?":** "I don't know. It hasn't been tested with real
students or teachers. The next step would be real, consented data with expert labels."
