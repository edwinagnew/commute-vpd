# Independent check: does the neuron set read whether the model already knows the trained fact?

*2026-11-06, independent check for baseline-lead. Read only, $0. Scripts and outputs in
`~/commute-data/scratch/baseline-lead/e107-check/`: `rederive.py` / `rederive.out` (overlaps and verdicts, stdlib only),
`tokens.py` / `tokens.out` (the true and wrong training texts, with the Qwen tokenizer), `cosines.py` / `cosines.out`
(cosines between saved updates). All through `cpu-run`; no model loaded.*

**Plain words.** I recomputed every overlap and verdict of the E107 score from the stored neuron sets, checked that the
wrong-answer fine-tune really trained on wrong answers, and compared the saved updates with each other directly. The
numbers and verdicts match exactly, the wrong answers are wrong in every row, and the wrong-answer update does pick
the same neurons as the true-answer update while staying as far from the fictional one. But in the weights the wrong
answers change the update only about as much as reshuffling the data does, and the probes' gains are the same whatever
facts were trained. So the test had little room to see familiarity, and says little about genuinely new facts.

**Verdict: holds with caveat.** The first two sentences of the claim hold as measured. "Topic alone moves the set" holds
on the TOFU probe only and cannot be told apart from answer length and starting loss. The last sentence, "no neuron
reading tried separates known from new facts at matched size", is the weakest: the wrong arm is a contradicted fact,
not a new one. At the matched step neither probe's gain depends on which facts were trained, and the label's effect on
the update is about the size of the seed noise the rule uses as its band. So the test had little power to read
familiarity.

## 1. Re-derivation from the stored sets

My own overlap and both rules, written from the plan text without importing `score.py`. Every input set holds 1,024
distinct neurons. The three stored references (x, L, the known seeds' overlap) re-derive from the stored sets to the
last digit. All 32 scored overlaps, the wrong-against-fictional overlaps and all four verdicts agree exactly with
`experiments/label_vs_style/results/score.json`.

| probe | rule | within-arm seeds | four cross pairs | band | line | verdict |
|---|---|---|---|---|---|---|
| TOFU (scored) | wrong against known | wrong 0.887, known 0.901 | 0.839 to 0.879 | 0.113 | topic or style needs ≥ 0.762 | topic or style |
| real (reported) | wrong against known | wrong 0.864, known 0.864 | 0.770 to 0.819 | 0.136 | ≥ 0.683 | topic or style |
| TOFU (scored) | authors against world | 0.879 / 0.896 | 0.728 to 0.739 | 0.121 | moves if under 0.754 | topic moves the set |
| real (reported) | authors against world | 0.799 / 0.858 | 0.600 to 0.624 | 0.201 | 0.617 | no call |

Wrong against fictional is 0.623 to 0.634 on TOFU and 0.504 to 0.508 on the real probe. Known against fictional is 0.614
to 0.637 and 0.470 to 0.492. So the wrong arm sits as far from the fictional arm as the known arm does, within 0.04.
Predictions were committed (27127ef6, 06:34) before the run and are unchanged since; the results commit is c3b86b3a,
06:55. The run log of job 53575 shows every guard passed: C2n reproduced exactly, W0's level exactly, the label check
held, and all six matched steps sit at 31 to 33.

The filed lean on the scored wrong-arm rule was "familiarity", and the run read "topic or style". That is a reversed
lean, and the analysis should say so plainly.

## 2. Did the wrong arm train on wrong answers, and learn them?

**The text is wrong.** `known_facts.swap` replaces every occurrence of the true answer with TOFU's
`perturbed_answer[0]` in pair 0's sentence and in the four other wordings, asserting the true answer is present and the
wrong one absent. `KEEP` protects "Boston Tea Party" and "Library of Alexandria", where the answer is part of an entity
name in the question. In no row does the wrong answer equal the true one. Spot checks read as intended, for example "The
author of the play 'Romeo and Juliet' is Charles Dickens." and "The Boston Tea Party happened in New York." The
questions are untouched, so the training prompts are byte-identical to the known arm's.

**The change is a small part of each text.** With the Qwen tokenizer, the stated answer is 2.6 of the 15.1 content
tokens of a wrong sentence, 17%. It carries 41% of W0's loss on the wrong sentence, against 16% on the true one. So
most of the wrong arm's training signal, 59% of its starting loss, sits on frame tokens it shares word for word with
the known arm.

**The label check holds.** W0 puts the wrong sentence below the true one in 186 of 217 rows, 0.857, mean gap 0.65 nats a
token.

**Something label-specific was learned, but not much.** At the matched step the wrong arm prefers the wrong sentence in
0.27 of rows, against 0.14 at W0. Its own wrong sentences rose by 1.92 nats a token; the true sentences of the same
rows rose by 1.39 to 1.40. The known arm, trained on those true sentences, raised them by 1.50 (E103 Amendments 1 and
2). So training on the wrong answers did 93% as much for the true sentences as training on the true answers did. The
mean gap between true and wrong sentences shrank from 0.65 to 0.12 to 0.13 nats a token. That shrink is not cleanly
label-specific: the known arm's matched state was never read on the wrong sentences, so how much of the shrink any update
of this size would give is unmeasured. The cleaner evidence that the arms learned different things is in the weights,
section 3: real, and about the size of seed noise.

**The probes do not see the facts at all.** The real probe is the 217 real questions with TOFU's true bare answers. Its
gain is 1.87 and 2.23 for the known arm, 2.02 and 2.07 for the wrong arm, and 2.18 and 2.45 for the fictional arm, which
never saw a real question. The TOFU probe gain is 1.69 to 1.79 for every E107 arm and the known arm. So whatever the
update did to the true answers of the trained questions, the probe's gain does not depend on whether those answers were
trained, contradicted or absent. The carrying set is ranked by contributions to that gain. A set ranked on a gain that
is the same for true, wrong and unrelated facts has little to read familiarity from.

## 3. Could the shared prompts or W0's gradient force the agreement?

**W0's gradient.** Every score multiplies the same W0 gradient on the probe. A Gaussian random change shares 0.21 to 0.25
of every set on the real probe and 0.26 to 0.28 on TOFU, the same for wrong, known, fictional and topic arms. That floor
raises every overlap by about the same amount; it cannot make wrong and known agree at 0.84 to 0.88 while both sit at
0.62 to 0.64 from the fictional arm.

**Shared prompts.** The wrong and known arms share every prompt and, at the same seed, the same batch order. Same-seed
pairs agree a little more than cross-seed ones: 0.869 and 0.879 against 0.839 and 0.860 on TOFU, 0.788 and 0.819 against
0.770 and 0.791 on the real probe. E105's two halves share no prompt and agree at 0.875 to 0.893 on TOFU and 0.818 to
0.842 on the real probe, as high as or higher than wrong against known. So shared prompts are not needed for this level
of agreement, and they do not produce it here beyond a few hundredths from data order.

**The updates themselves.** Cosine between saved matched updates, over all 196 weight matrices (MLP matrices alone read
within 0.02 of these):

| pair | same seed | different seed |
|---|---|---|
| wrong against wrong | | 0.837 |
| known against known | | 0.821 |
| wrong against known | 0.825, 0.821 | 0.725, 0.728 |
| authors against authors | | 0.904 |
| world against world | | 0.867 |
| authors against world | 0.398, 0.418 | 0.407, 0.409 |
| authors or world against known | | 0.695 to 0.717 |

So in weight space the label does change the update. With the data order fixed, swapping true for wrong answers turns
the update by as much as changing the data order does: 0.82 either way. Changing both gives 0.73. So the label's effect
is about the size of seed noise, and the filed rule's band is seed noise. The rule could therefore only have called
"familiarity" for an effect several times larger than the one the weights show. The set reads the same small effect:
same-seed pairs 0.87 and 0.88 against cross-seed 0.84 and 0.86 on TOFU. This is the one number that shows the two arms
learned different things, and it shows the difference is real but small.

**What that implies.** The wrong arm differs from the known arm only in what is said per row. E105 already showed the set
does not move when every row is replaced, with label and style held. The familiarity manipulation lives in exactly the
per-row part E105 found the set ignores at this step. So "wrong sits with known" was close to implied by E105 before the
run. It is a real reading, but it shows that at this size the set reads the shared style direction of an update, not the
facts in it, rather than adding an independent line of evidence against a familiarity reading.

## 4. Could "topic moves the set" come from row counts or answer lengths?

| split | rows | content tokens a sentence | W0 level, nats a token | epochs at the matched step | own gain | other side's gain |
|---|---|---|---|---|---|---|
| authors | 100 | 17.7 | −1.75 | 0.99 | 1.37 to 1.38 | 1.52 to 1.53 |
| world facts | 117 | 13.0 | −2.24 | 0.88 to 0.90 | 1.64 to 1.71 | 1.21 to 1.27 |
| E105 half A | 108 | 15.3 | −2.10 | 0.95 | | |
| E105 half B | 109 | 15.0 | −1.93 | 0.95 | | |

**Row counts: unlikely.** 100 and 117 are within 10% of E105's 108 and 109, and E105 found 108 against 217 rows did not
move the set (halves against the full arm 0.83 to 0.86 on the real probe). Exposure per row is within 0.1 epoch of E105's.

**Answer length and starting loss: not separable.** Author sentences are 36% longer than world-fact sentences and start
0.49 nats a token more likely under W0, so they have less room to gain. The author arms barely clear the 1.24 content bar and gain more
on the world rows than on their own. The topic split moves sentence length, W0's starting level and entity type together
with topic, and the run cannot say which of them moves the set.

**It is mostly the author arms that move.** On TOFU, world against known reads 0.83 to 0.85, close to L at 0.875. Author
against known reads 0.77 to 0.79, and author against E105's half A 0.78 to 0.80. On the real probe the same: world against
known 0.74 to 0.80, author 0.64 to 0.71. Author arms are also further from the fictional sets, 0.59 against 0.64 on TOFU.
So the fictional author biographies are not closer to the real author arm, which argues against "entity type" as the
driver. On the real probe the rule reads no call, with the author arm's seeds at only 0.80.

**In the weights the split is large.** Author against world updates have cosine 0.40 to 0.42, against 0.87 to 0.90
between seeds. That is far below wrong against known at 0.73. But E105's random halves were not saved, so whether any
two disjoint halves of these rows give updates this far apart is unknown. The weight cosine cannot yet say "topic"
rather than "different rows".

## 5. What the claim can say

Supported: at matched update size, a fine-tune on wrong answers about the same real people and places, in the same
words, picks nearly the same 1,024 carrying neurons as the fine-tune on the true answers, on both probes, and stays as far
from the fictional-facts fine-tune as the true one does. Splitting the true rows by source file moves the set on the TOFU
probe beyond seed noise.

Not supported as written:
- "topic alone": the split also changes sentence length and starting loss, and the real probe reads no call. Say "the
  source-file split moves the set on the TOFU probe".
- "no neuron reading tried separates known from new facts": the wrong arm is a contradicted fact, which the plan's own
  caveat says is not a new one. And at the matched step the probes' gains are the same whatever facts were trained, so a
  familiarity effect had little to show through. A fair line: "at matched size, set identity does not read whether the
  trained answer agrees with what the model already says; what it separates is the data's source and style. Whether
  genuinely new facts move it is untested."

Weakest sub-claim: the closing generalisation to "new facts".

## Not run

- The known arm's matched state read on the wrong sentences: the control that would say whether the wrong arm's
  shrinking true-minus-wrong gap is label-specific or what any update of this size does. Seconds on a GPU; the states are
  saved.
- A read at a later step or larger size, where the arms have learned their own facts.
- A per-row or answer-token-only carrying read, ranked on the answer-span tokens alone.
- A random split at the topic arms' row counts and lengths, or a topic split matched on sentence length. Saving its
  updates would give the weight-cosine reference the topic reading lacks.
- A label rule whose band is the measured label effect rather than seed noise, or more seeds to shrink the band.
- Real facts W0 lacks (option 3) or invented names on the author rows (option 6).
- VPD reads of the wrong-arm states (qwen-decomp-lead).
