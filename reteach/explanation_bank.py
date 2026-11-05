"""Hand-written explanation templates used to build SYNTHETIC data.

Structure
---------
``FAMILIES[label]`` is a list of *families*. Each family is one distinct line
of reasoning, written as two paraphrases. Paraphrases of the same family share
a ``group_id`` and are always placed in the same train/test split, so the test
set never contains a reworded copy of a training sentence.

Placeholders such as ``{a}`` or ``{lcd}`` are filled from a question context
(see ``reteach.datagen.context_values``). The numbers differ between
development and held-out contexts, but number substitution is NOT counted as
new evidence: each paraphrase is used at most once in any experiment.

``DEMO_EXPLANATIONS`` is a *separate* pool used only for the fictional class
in the demo app, so app predictions are made on wording the classifier never
trained on. All of this text was written by the project author; none of it
comes from real students.
"""

from __future__ import annotations

from . import taxonomy as tx

ADD, EQUIV, COMPARE = "add", "equivalent", "compare_larger"

# (operation, [paraphrase, paraphrase])
FAMILIES: dict[str, list[tuple[str, list[str]]]] = {
    tx.CORRECT: [
        (ADD, ["I changed both fractions to {lcd}ths. {a} is {sa}/{lcd} and {b} is {sb}/{lcd}, "
               "so the answer is {ans}.",
               "Both need a denominator of {lcd}: {a} becomes {sa}/{lcd}, {b} becomes {sb}/{lcd}, "
               "then add the tops to get {ans}."]),
        (ADD, ["The pieces have to be the same size before you add, so I rewrote both with {lcd} "
               "as the denominator and then added.",
               "you cant add different sized pieces. i made them the same size pieces first then "
               "added them"]),
        (ADD, ["I multiplied the top and bottom of {a} by {fa} and the top and bottom of {b} by "
               "{fb}, then added the numerators and kept the denominator {lcd}.",
               "Multiply numerator and denominator by the same number for each fraction so they "
               "share a denominator, then add only the numerators."]),
        (ADD, ["I found the least common multiple of {da} and {db}, which is {lcd}, used it as the "
               "common denominator and added the equivalent fractions.",
               "LCM of the bottoms is {lcd}. Convert each fraction to an equal fraction over {lcd} "
               "and add."]),
        (ADD, ["My estimate was a bit more than {a}, and {ans} makes sense because I converted to a "
               "common denominator before adding.",
               "I checked that the answer is bigger than both fractions, which it has to be when "
               "you add. I used a common denominator to get it."]),
        (EQUIV, ["To get from {da} to {target} you multiply by {factor}, so you multiply the top by "
                 "{factor} too: {ans}.",
                 "{da} times {factor} is {target}, so {na} times {factor} is {new_n}. Same "
                 "multiplier top and bottom."]),
        (EQUIV, ["Whatever you multiply the denominator by you have to multiply the numerator by "
                 "the same amount, so the fraction stays the same size.",
                 "you have to times the top and bottom by the same number, {factor}, so its still "
                 "the same amount"]),
        (EQUIV, ["I drew {a} and split each part into {factor} smaller pieces. Now there are "
                 "{target} pieces and {new_n} are shaded.",
                 "If you cut every piece into {factor} pieces, the shaded part is {new_n} of "
                 "{target} pieces, so it is the same amount."]),
        (EQUIV, ["{ans} simplifies back to {a} when you divide top and bottom by {factor}, so they "
                 "are equivalent.",
                 "I checked by dividing {new_n} and {target} by {factor}, which gives {a} again."]),
        (COMPARE, ["{big} is bigger because the whole is cut into fewer pieces, so each piece is "
                   "larger.",
                   "Fewer pieces means each piece is bigger, so {big} is more than {small}."]),
        (COMPARE, ["Both have {n} on top, so the one with the smaller denominator is bigger. It is "
                   "{big}.",
                   "Same numerators, so compare the denominators: a smaller denominator gives a "
                   "larger fraction."]),
        (COMPARE, ["If you share a pizza between {dfew} people you get more than sharing it "
                   "between {dmany} people, so {big} is larger.",
                   "Sharing with fewer people gives everyone a bigger slice, that's why {big} is "
                   "more."]),
    ],
    tx.ADD_ACROSS: [
        (ADD, ["I added the numerators {na} + {nb} = {sum_n} and the denominators {da} + {db} = "
               "{sum_d}, so {across}.",
               "top plus top is {sum_n} and bottom plus bottom is {sum_d}, so I got {across}"]),
        (ADD, ["You add the tops together and the bottoms together.",
               "just add straight across, numerators then denominators"]),
        (ADD, ["Adding fractions is like adding two numbers, so I added both parts.",
               "I treated the top numbers and bottom numbers as two separate sums."]),
        (ADD, ["{na} plus {nb} is {sum_n}, {da} plus {db} is {sum_d}. Answer {across}.",
               "I did {na}+{nb} on top and {da}+{db} on the bottom."]),
        (ADD, ["Since it's an addition problem I added everything: the numerators and the "
               "denominators.",
               "It says plus so I added all of the numbers in the fractions."]),
        (ADD, ["I put the sum of the numerators over the sum of the denominators.",
               "The new fraction is the numerators added over the denominators added."]),
        (ADD, ["I lined up the fractions and added the top row and the bottom row.",
               "Add the top row, then add the bottom row, and that is the fraction."]),
        (ADD, ["It's like multiplying fractions where you go straight across, but with plus.",
               "When you multiply fractions you go across, so for adding I also went across."]),
        (ADD, ["You don't need a common denominator, you just add the denominators too.",
               "no need to change anything, the bottoms get added like the tops"]),
        (ADD, ["I combined {da} pieces and {db} pieces to get {sum_d} pieces and {sum_n} of them "
               "are shaded.",
               "There are {sum_d} pieces altogether and {sum_n} are shaded so it's {across}."]),
        (ADD, ["the answer is {sum_n} over {sum_d}",
               "i got {sum_n} on top and {sum_d} on bottom"]),
        (ADD, ["Add the parts and add the wholes.",
               "I added the parts together and the wholes together to get the total."]),
    ],
    tx.UNSCALED_NUMERATOR: [
        (ADD, ["I made the denominators both {lcd} and then added {na} + {nb} to get {unscaled}.",
               "Common denominator is {lcd}, then {na} plus {nb} is {sum_n}, so {unscaled}."]),
        (ADD, ["I changed the bottoms to be the same but kept the tops the same.",
               "only the denominators need to match so I didn't change the numerators"]),
        (ADD, ["I multiplied the denominators to get {prod} and added the numerators to get "
               "{sum_n}.",
               "Bottom numbers times each other gives {prod}, then add the top numbers."]),
        (ADD, ["The denominator becomes {lcd} because both go into it. The numerators stay {na} "
               "and {nb}.",
               "Both go into {lcd}, so that is the denominator. The numerators don't change."]),
        (ADD, ["You only change the denominator to the common one, the numerator stays as it is.",
               "Change the denominator, keep the numerator."]),
        (ADD, ["I rewrote {a} as {na}/{lcd} and {b} as {nb}/{lcd}, then added.",
               "{a} turned into {na}/{lcd} and {b} turned into {nb}/{lcd}."]),
        (ADD, ["I found the common denominator and put the numerators on top of it.",
               "found the LCD and then added the numerators over it"]),
        (EQUIV, ["The new denominator is {target} and the numerator stays {na}, so {na}/{target}.",
                 "I put {na} over {target} because the top doesn't change."]),
        (EQUIV, ["You just change the bottom number to {target}.",
                 "only the bottom number changes, the top is the same"]),
        (EQUIV, ["I multiplied the denominator by {factor} to get {target}. The numerator is still "
                 "{na}.",
                 "{da} times {factor} is {target}, numerator stays {na}"]),
        (EQUIV, ["Equivalent means same numerator, different denominator.",
                 "An equivalent fraction keeps the same top number and has a new bottom number."]),
        (EQUIV, ["The question gives the new denominator so I just wrote it under the old "
                 "numerator.",
                 "I kept the numerator and wrote the denominator they asked for."]),
    ],
    tx.ADDITIVE_EQUIVALENCE: [
        (EQUIV, ["I added {diff} to the bottom to get {target}, so I added {diff} to the top too: "
                 "{additive}.",
                 "{da} plus {diff} is {target}, and {na} plus {diff} is {add_n}."]),
        (EQUIV, ["Whatever you add to the bottom you add to the top.",
                 "you have to add the same amount to the numerator and denominator"]),
        (EQUIV, ["The denominator went up by {diff} so the numerator goes up by {diff}.",
                 "Bottom went up {diff}, so top goes up {diff} as well."]),
        (EQUIV, ["It's {diff} more on the bottom, so it has to be {diff} more on the top to stay "
                 "equal.",
                 "To keep it fair I added {diff} to both numbers."]),
        (EQUIV, ["I counted up from {da} to {target}, that's {diff}, then counted up {diff} from "
                 "{na}.",
                 "Counting on from {da} to {target} is {diff} steps, so I counted on {diff} steps "
                 "from {na} too."]),
        (EQUIV, ["Doing the same thing to the top and bottom keeps the fraction the same, so I "
                 "added {diff} to both.",
                 "If you do the same to both numbers it stays equal, so plus {diff} on each."]),
        (EQUIV, ["The difference between the numerator and denominator stays the same in "
                 "equivalent fractions.",
                 "In {a} the gap between top and bottom is {gap}, so in the new fraction the gap "
                 "is still {gap}."]),
        (EQUIV, ["{additive} because I added {diff} to {na} and to {da}.",
                 "plus {diff} on top and plus {diff} on bottom gives {additive}"]),
        (EQUIV, ["I need to get to {target}, so I add the difference to both parts.",
                 "find how much you need to add to get the new denominator and add that to both"]),
        (EQUIV, ["You add the same number so the fraction doesn't change.",
                 "adding the same number to both parts keeps it equal"]),
        (EQUIV, ["the top and bottom both go up by the same amount",
                 "both numbers increase by the same amount"]),
        (EQUIV, ["I balanced it like an equation by adding {diff} to each part of the fraction.",
                 "Like balancing, I added {diff} to the top and the bottom."]),
    ],
    tx.LARGER_DENOMINATOR: [
        (COMPARE, ["{small} is bigger because {dmany} is bigger than {dfew}.",
                   "{dmany} is more than {dfew}, so {small} is the larger fraction."]),
        (COMPARE, ["The bigger the denominator, the bigger the fraction.",
                   "a bigger number on the bottom means a bigger fraction"]),
        (COMPARE, ["The numerators are the same so I looked at the denominators and picked the "
                   "bigger one.",
                   "Same top, so the one with the bigger bottom number is bigger."]),
        (COMPARE, ["{small} has more pieces so it is more.",
                   "There are {dmany} pieces in {small} and only {dfew} in {big}, so {small} is "
                   "more."]),
        (COMPARE, ["{dmany} is a larger number, so {small} is larger.",
                   "I compared the numbers and {dmany} is larger."]),
        (COMPARE, ["More parts means more of the whole.",
                   "having more parts makes the fraction bigger"]),
        (COMPARE, ["You compare fractions like whole numbers, so look at which bottom is bigger.",
                   "I compared them like normal numbers"]),
        (COMPARE, ["{small} because it comes later when you count: {dfew}, then {dmany}.",
                   "{dmany} comes after {dfew} when counting, so it's bigger."]),
        (COMPARE, ["If you cut it into {dmany} pieces you get more pizza than cutting it into "
                   "{dfew}.",
                   "cutting into more slices gives you more pizza"]),
        (COMPARE, ["The denominator tells you how big it is, so the bigger denominator wins.",
                   "the bottom number shows the size and {dmany} is bigger"]),
        (COMPARE, ["{small} is greater since its denominator is greater.",
                   "its denominator is greater so the fraction is greater"]),
        (COMPARE, ["The larger number on the bottom makes it worth more.",
                   "a large bottom number is worth more"]),
    ],
}

# Separate wording used ONLY for the demo class (never for training/testing).
DEMO_EXPLANATIONS: dict[tuple[str, str], list[str]] = {
    (tx.CORRECT, ADD): [
        "I turned them into fractions with the same denominator {lcd} and then added the tops.",
        "Rewrite {a} and {b} so they have the same denominator, then add numerators.",
        "first make equal size pieces using {lcd}, then add how many pieces",
    ],
    (tx.CORRECT, EQUIV): [
        "multiply both the top and bottom by {factor}",
        "{na} times {factor} and {da} times {factor} gives the same fraction",
    ],
    (tx.CORRECT, COMPARE): [
        "{big} because the pieces are bigger when there are fewer of them",
        "fewer slices so each slice is bigger",
    ],
    (tx.ADD_ACROSS, ADD): [
        "I added the top numbers and then the bottom numbers",
        "{na}+{nb} and {da}+{db}",
        "add numerator with numerator and denominator with denominator",
    ],
    (tx.UNSCALED_NUMERATOR, ADD): [
        "made the bottom {lcd} and added the tops",
        "same denominator then add the numerators {na} and {nb}",
    ],
    (tx.UNSCALED_NUMERATOR, EQUIV): [
        "the top stays {na} and the bottom is {target}",
        "I only changed the denominator",
    ],
    (tx.ADDITIVE_EQUIVALENCE, EQUIV): [
        "I added {diff} to both the top and the bottom",
        "the bottom needs {diff} more so the top gets {diff} more",
    ],
    (tx.LARGER_DENOMINATOR, COMPARE): [
        "{dmany} is bigger than {dfew}",
        "the one with the bigger denominator is bigger",
        "it has more pieces",
    ],
}
# Wording for wrong answers that do not match a mapped misconception, plus
# vague answers. These are expected to land in insufficient/review states.
DEMO_VAGUE = [
    "I guessed",
    "idk",
    "I think this one is right",
    "my friend said so",
    "it looked right to me",
    "because of the numbers",
]

# Additional families (added in iteration 2 to increase the number of
# independent reasoning patterns per class; see docs/evaluation.md).
FAMILIES[tx.CORRECT] += [
    (ADD, ["I used a number line split into {lcd} parts and counted {sa} parts then {sb} more "
           "parts.",
           "On a number line marked in {lcd}ths, I jumped {sa} and then {sb}, landing on {ans}."]),
    (ADD, ["The sum should be less than 1 here, and {ans} is, so I trust converting to "
           "{lcd}ths.",
           "Sanity check: each fraction is under one half or close, so the total is reasonable. I "
           "used equivalent fractions with denominator {lcd}."]),
    (ADD, ["Equivalent fractions first, then add. {a} = {sa}/{lcd}, {b} = {sb}/{lcd}.",
           "I renamed each fraction as an equal fraction in {lcd}ths and then combined them."]),
    (EQUIV, ["I scaled {a} up by a factor of {factor}, which is the same as multiplying by "
             "{factor}/{factor} = 1.",
             "Multiplying by {factor}/{factor} is multiplying by one, so the value doesn't change."]),
    (EQUIV, ["I used a multiplication table: the {da} row and the {na} row, go across to {target} "
             "and read {new_n} above it.",
             "Looking at the times table, {da} times {factor} is {target} and {na} times {factor} "
             "is {new_n}."]),
    (EQUIV, ["The ratio of top to bottom has to stay the same, so both get scaled by {factor}.",
             "Keeping the same proportion means multiplying both parts by {factor}."]),
    (COMPARE, ["On a number line {big} is further from zero than {small}.",
               "I placed both on a number line from 0 to 1 and {big} is further right."]),
    (COMPARE, ["{dmany}ths are smaller pieces than {dfew}ths, and both have {n} pieces, so {big} "
               "is larger.",
               "Each {dmany}th is tiny compared to a {dfew}th, so the same number of pieces is "
               "worth less."]),
]
FAMILIES[tx.ADD_ACROSS] += [
    (ADD, ["{across} because you combine the numerators and you combine the denominators.",
           "combine numerators, combine denominators"]),
    (ADD, ["The denominators are different so I just added them like the numerators.",
           "Different bottoms, so I added the bottoms as well."]),
    (ADD, ["I added {a} and {b} like two scores: {na} out of {da} and {nb} out of {db} makes "
           "{sum_n} out of {sum_d}.",
           "Like test scores, {na} out of {da} plus {nb} out of {db} is {sum_n} out of {sum_d}."]),
    (ADD, ["Numerator plus numerator equals the new numerator and denominator plus denominator "
           "equals the new denominator.",
           "new top = top + top, new bottom = bottom + bottom"]),
    (ADD, ["I just did the plus on both levels of the fraction.",
           "Did the addition on the top level and the bottom level."]),
    (ADD, ["{sum_n}/{sum_d} since {na} and {nb} make {sum_n} and {da} and {db} make {sum_d}.",
           "{na} and {nb} make {sum_n}; {da} and {db} make {sum_d}."]),
    (ADD, ["Both fractions get added the same way as whole numbers, top with top and bottom "
           "with bottom.",
           "I added them like whole numbers, tops together and bottoms together."]),
    (ADD, ["I added the two fractions straight across without changing them first.",
           "Didn't change anything, just added across."]),
]
FAMILIES[tx.UNSCALED_NUMERATOR] += [
    (ADD, ["I made both fractions out of {lcd} and kept {na} and {nb} on top, giving "
           "{unscaled}.",
           "Out of {lcd}, the tops are still {na} and {nb}, so {unscaled}."]),
    (ADD, ["The denominators have to be the same to add, so I made them {lcd}. Then {na} + "
           "{nb}.",
           "Made the denominators equal at {lcd}, then added the original numerators."]),
    (ADD, ["I multiplied the bottom numbers by each other but the top numbers just get added.",
           "The bottoms get multiplied together and the tops get added."]),
    (ADD, ["The numerator counts how many pieces there are, so {na} piece and {nb} pieces make "
           "{sum_n} pieces of size {lcd}.",
           "{na} piece plus {nb} pieces is {sum_n} pieces, and I wrote them over {lcd}."]),
    (ADD, ["Found the common denominator {lcd} and left the numerators alone.",
           "left the numerators alone after finding the common denominator"]),
    (EQUIV, ["The numerator doesn't need to change when the denominator changes to {target}.",
             "Changing the denominator to {target} doesn't change the numerator."]),
    (EQUIV, ["I just replaced {da} with {target}, so it's {na}/{target}.",
             "Swapped the bottom {da} for {target} and kept {na}."]),
    (EQUIV, ["It's the same number of pieces, {na}, just out of {target} now.",
             "Still {na} pieces, but out of {target}."]),
]
FAMILIES[tx.ADDITIVE_EQUIVALENCE] += [
    (EQUIV, ["{target} minus {da} is {diff}, so the numerator is {na} plus {diff}.",
             "I subtracted {da} from {target} to get {diff} and added it to {na}."]),
    (EQUIV, ["Both numbers move up by {diff} so it is the same fraction.",
             "if both move up {diff} it's still the same fraction"]),
    (EQUIV, ["To make an equivalent fraction you add the same thing to the top and bottom.",
             "Equivalent fractions come from adding equal amounts to the numerator and "
             "denominator."]),
    (EQUIV, ["Adding {diff} to the bottom made it {target}, so adding {diff} to {na} makes "
             "{add_n}.",
             "plus {diff} gets the bottom to {target} and plus {diff} gets the top to {add_n}"]),
    (EQUIV, ["The top is always {gap} less than the bottom, so it's {add_n}/{target}.",
             "Numerator stays {gap} below the denominator."]),
    (EQUIV, ["I added on to both parts by the same number so it balances.",
             "Same amount added on to each part keeps it balanced."]),
    (EQUIV, ["You get from {da} to {target} by adding, so do that addition to the numerator.",
             "Add on to the numerator whatever you added to the denominator."]),
    (EQUIV, ["I increased the numerator and denominator by the same number.",
             "increase both by the same number"]),
]
FAMILIES[tx.LARGER_DENOMINATOR] += [
    (COMPARE, ["{small} is bigger since {dmany} is a higher number.",
               "Higher number on the bottom, so {small} is more."]),
    (COMPARE, ["You look at the denominators and the larger one is the larger fraction.",
               "look at the bottoms, the larger bottom is larger"]),
    (COMPARE, ["{dmany} pieces is more than {dfew} pieces, so {small} has more.",
               "more pieces is more than fewer pieces"]),
    (COMPARE, ["Since the tops are both {n}, the bottom decides, and {dmany} beats {dfew}.",
               "Tops match, so the bigger bottom number wins."]),
    (COMPARE, ["{dmany} is greater than {dfew} like on a normal number line, so {small} is "
               "greater.",
               "On a number line {dmany} is past {dfew}, so {small} is the bigger one."]),
    (COMPARE, ["{small} because you are getting {dmany} slices instead of {dfew}.",
               "you get more slices with {small}"]),
    (COMPARE, ["The fraction with the biggest number in it is the biggest fraction.",
               "biggest number means biggest fraction"]),
    (COMPARE, ["A bigger denominator makes the fraction worth more.",
               "bigger denominator = worth more"]),
]
