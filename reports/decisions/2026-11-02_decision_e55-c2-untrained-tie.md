# Decision: run the long VPD decomposition on the counting set, now that a barely trained one already scores perfectly?

**Reuse-lead, 2026-11-02. Waits on the user; nothing long runs until it is answered.**

A $0 cluster check put VPD's read and score code through a decomposition trained for 400 of its 400,000 steps. That
decomposition already labels all 21 scored fine-tunes correctly under the filed rule, so a perfect score on this set
does not need a trained decomposition. We recommend dropping the 56-hour run here and scoring VPD on fine-tunes where a
decomposition exists and methods can disagree.

## What was seen

- **The check.** Job 48413 read all 43 states on the 400-step decomposition, reported only, never scored as a result. All 21 are
  labelled right on both readings, the outcome the filing calls a tie.
- **How small the numbers are.** The fine-tunes that reuse the old count read near 0 (−0.006 to +0.005) and the ones that
  learn a new count read near −0.05 (−0.058 to −0.037), with the cut-off at −0.018. On the 67M, the elicit and cipher
  fine-tunes read 0.77 and 0.24, so this gap is far too small to be the effect VPD is supposed to measure.
- **The control caught it.** The identity-prompt fine-tunes, which must reach 0.5, read about 0.07, so the positive control
  says "none", as it should for an untrained decomposition. Today that changes no outcome.
- **Why it separates: not the answer token.** Fable expected the gap to follow the token being read out. Job 48441 tested
  that on two fine-tunes of each kind, and the gap follows the fine-tune instead: the two new-count fine-tunes read −0.87 and
  −1.07 at the other kind's token, against −0.19 and −0.01 for the two reuse fine-tunes. So a nearly untrained importance
  function already responds more to the new-count updates. Why it does is not established; four fine-tunes on one
  decomposition cannot say.
- **Other readings.** At least 0.68 of every fine-tune's gain sits in the decomposed matrices, so none is void. Reading all
  43 takes 21 minutes on 2.8 GiB. The cut that keeps the answer token from the importance function moved no state by over 0.1 at this step.

## The options

**A. Drop this run (recommended).** Nothing runs. VPD gets no score on the counting set, so the brief's "score VPD against
a ground truth" stays open there. What we learn is already in hand: on this set VPD can at best tie the cheap methods, and a
tie comes free. Next, a plan to score VPD on the 67M and Qwen fine-tunes, whose decompositions exist, where a VPD label can
disagree with the others.

**B. Run one 24-hour job, then read.** About 170,000 steps on one shared L40, $0. It tells us whether a decomposition of this
small model trains at all: the identity-prompt reading and how close the masked model gets to the real one. Worth it only if we
want a trained decomposition of a small model with known ground truth for later work. The next two jobs go ahead only if an
identity-prompt reading reaches 0.25 or the masked model's error halves (Fable's bars, not filed).

**C. Run the whole thing, after amending the rule.** Three 24-hour jobs, about 2.5 days of one of 8 shared L40s, $0. The rule
changes first so that a tie counts only if the identity-prompt control holds; otherwise it is written "tie, apparatus
unverified". Most likely it ends in a tie that adds little to the untrained result. The one informative outcome, VPD
mislabelling a fine-tune every cheap method gets right with the control holding, Fable puts at about 15%.

## Why A

Fable's advice (`reports/plans/2026-11-02_advice_c2-untrained-tie-FABLE.md`) recommends A; its reasoning leans on the token
explanation that job 48441 did not support, but the conclusion needs only the tie itself, which stands. What would change it: if the
user wants a trained small decomposition for its own sake (then B), or if VPD's result on this exact set matters to a reader
regardless of the tie (then C).

## Not run

The full decomposition; any reading on a trained decomposition; the token and update comparison beyond four fine-tunes; a
reading at any step between 400 and 400,000.

## Departures from the literature

The checks here use the plan's recipe at 400 of its 400,000 steps, which no source recommends for reading; these readings are a negative
control, not a VPD result.
