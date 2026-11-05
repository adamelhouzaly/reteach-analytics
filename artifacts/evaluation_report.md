# Explanation classifier — evaluation report (auto-generated)

> ALL DATA IS SYNTHETIC: author-written template explanations, not real student work. Results do not establish classroom effectiveness.

Seed 42. Dev: 130 examples / 65 families. Test: 70 examples / 35 families (held-out families AND held-out question contexts C-ADD-5, C-ADD-6, C-CMP-4, C-CMP-5, C-EQ-4, C-EQ-5). Family overlap dev/test: 0.

## Headline (test split)

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression (C=0.3) | 64.3% | 0.616 |
| Most-frequent baseline (predicts `add_across`) | 20.0% | 0.067 |

## Per-class (test split)

| Class | Precision | Recall | F1 | Support (examples) | Test families |
|---|---|---|---|---|---|
| correct_reasoning | 0.43 | 0.43 | 0.43 | 14 | 7 |
| add_across | 0.61 | 0.79 | 0.69 | 14 | 7 |
| unscaled_numerator | 0.60 | 0.21 | 0.32 | 14 | 7 |
| additive_equivalence | 0.67 | 0.86 | 0.75 | 14 | 7 |
| larger_denominator_bigger | 0.87 | 0.93 | 0.90 | 14 | 7 |

## Confusion matrix (rows = true, columns = predicted)

| true \ pred | correct_reasoning | add_across | unscaled_numerator | additive_equivalence | larger_denominator_bigger |
|---|---|---|---|---|---|
| correct_reasoning | 6 | 1 | 1 | 4 | 2 |
| add_across | 2 | 11 | 1 | 0 | 0 |
| unscaled_numerator | 6 | 3 | 3 | 2 | 0 |
| additive_equivalence | 0 | 2 | 0 | 12 | 0 |
| larger_denominator_bigger | 0 | 1 | 0 | 0 | 13 |

## Review threshold

Chosen on dev out-of-fold predictions (lowest threshold with validation accepted-accuracy >= 90%): **1.01**. On test: coverage 0.0% (0 auto-accepted, 70 routed to review); accuracy on accepted n/a; accuracy on routed 64.3%. Scores are uncalibrated.

## Misclassified test examples (highest score first)

| Explanation | True | Predicted | Score |
|---|---|---|---|
| The ratio of top to bottom has to stay the same, so both get scaled by 3. | correct_reasoning | additive_equivalence | 0.26 |
| The bottoms get multiplied together and the tops get added. | unscaled_numerator | add_across | 0.26 |
| Bottom went up 10, so top goes up 10 as well. | additive_equivalence | add_across | 0.25 |
| I used a number line split into 14 parts and counted 4 parts then 7 more parts. | correct_reasoning | larger_denominator_bigger | 0.25 |
| I compared them like normal numbers | larger_denominator_bigger | add_across | 0.24 |
| I multiplied the denominators to get 15 and added the numerators to get 2. | unscaled_numerator | add_across | 0.24 |
| To get from 5 to 15 you multiply by 3, so you multiply the top by 3 too: 6/15. | correct_reasoning | additive_equivalence | 0.24 |
| I multiplied the bottom numbers by each other but the top numbers just get added. | unscaled_numerator | add_across | 0.24 |
| I rewrote 1/3 as 1/15 and 1/5 as 1/15, then added. | unscaled_numerator | correct_reasoning | 0.24 |
| Common denominator is 15, then 1 plus 1 is 2, so 2/15. | unscaled_numerator | correct_reasoning | 0.24 |
| Keeping the same proportion means multiplying both parts by 3. | correct_reasoning | additive_equivalence | 0.23 |
| The pieces have to be the same size before you add, so I rewrote both with 14 as the denominator and then added. | correct_reasoning | unscaled_numerator | 0.23 |
| Bottom numbers times each other gives 15, then add the top numbers. | unscaled_numerator | correct_reasoning | 0.23 |
| I multiplied the denominator by 3 to get 15. The numerator is still 2. | unscaled_numerator | additive_equivalence | 0.23 |
| Didn't change anything, just added across. | add_across | unscaled_numerator | 0.23 |
| I drew 2/5 and split each part into 3 smaller pieces. Now there are 15 pieces and 6 are shaded. | correct_reasoning | add_across | 0.22 |
| the answer is 3 over 9 | add_across | correct_reasoning | 0.22 |
| I made the denominators both 14 and then added 2 + 1 to get 3/14. | unscaled_numerator | additive_equivalence | 0.22 |
| My estimate was a bit more than 1/3, and 8/15 makes sense because I converted to a common denominator before adding. | correct_reasoning | larger_denominator_bigger | 0.22 |
| 3 times 4 is 12, numerator stays 1 | unscaled_numerator | correct_reasoning | 0.22 |
| 15 minus 5 is 10, so the numerator is 2 plus 10. | additive_equivalence | add_across | 0.22 |
| 1 piece plus 1 pieces is 2 pieces, and I wrote them over 15. | unscaled_numerator | correct_reasoning | 0.22 |
| When you multiply fractions you go across, so for adding I also went across. | add_across | correct_reasoning | 0.22 |
| 4/12 simplifies back to 1/3 when you divide top and bottom by 4, so they are equivalent. | correct_reasoning | additive_equivalence | 0.21 |
| The numerator counts how many pieces there are, so 1 piece and 1 pieces make 2 pieces of size 15. | unscaled_numerator | correct_reasoning | 0.21 |
