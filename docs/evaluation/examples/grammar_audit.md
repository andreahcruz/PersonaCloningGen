# Grammar tool audit

Base text: 600 two sentence windows of real Lemkin writing that the tool finds clean.

## 1. Rule recall (injected errors of the types the tool has rules for)

| Rule | Injected | Caught | Recall |
|---|---|---|---|
| repeated_word | 60 | 44 | 73% |
| missing_space_after_punct | 60 | 60 | 100% |
| space_before_punct | 60 | 60 | 100% |
| lowercase_sentence_start | 60 | 56 | 93% |
| lowercase_i | 56 | 55 | 98% |
| double_punct | 60 | 60 | 100% |
| a_an | 60 | 60 | 100% |
| missing_apostrophe | 6 | 6 | 100% |
| modal_of | 8 | 8 | 100% |
| agreement | 39 | 22 | 56% |
| misspelling | 60 | 60 | 100% |
| broken_character | 23 | 23 | 100% |

Why two rules are below 100 percent. These are deliberate choices to cut false alarms on real posts, not bugs. `repeated_word` ignores repeats where both words are capitalized ("Go Go Days"), because those are usually names or titles. `agreement` ignores "you is" and "I are", because "what I can tell you is this" and "Jason and I are" are correct English. The injected errors include those cases, so the recall shows what the tool gives up for precision. Recall on the other rules is high partly by construction, because the injected errors are exactly the patterns the rules look for.

## 2. Coverage of ordinary grammar mistakes

The tool flags 0 of 40 hand written mistakes (0%).
On the sanity set of 4 mistakes the rules are written for, it flags 4.

| Category | Mistakes | Flagged |
|---|---|---|
| subject_verb_agreement | 4 | 0 |
| verb_tense | 4 | 0 |
| confused_words | 4 | 0 |
| missing_words | 4 | 0 |
| preposition | 4 | 0 |
| comma_splice_run_on | 4 | 0 |
| pronoun_case | 4 | 0 |
| plural_countability | 4 | 0 |
| double_negative | 4 | 0 |
| awkward_word_choice | 4 | 0 |

Read this as: the rules do what they say (section 1), and they cover only a small part of grammar (section 2).
So the tool is a floor check for mechanical slips, not a grammar score. A stronger checker is needed for more.
