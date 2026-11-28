# Check of E331's results, the scaled adjoint split on the built edits (review, 2026-11-27)

I re-read E331's saved result lines and checked regating-lead's numbers and proposed paper line against them. Every
number in their message matches the files. The scaled split puts the seven edits in the same order as least squares,
at lower values. The proposed sentence is right about the scaled split. Its last clause, about the split without its
scale, is wrong and needs rewording. One filed rival is also met as written, which the page should say.

Sources: `experiments/adjoint_rho/results/items_run2_s{0..3}.jsonl` (E331) and `items_s{0..3}.jsonl` (E329), at main
41b087a3.

## Numbers checked

| Item | Least squares (E268) | Unscaled (E329) | Scaled (E331) | Scaled, over E268's needed set | c (min / median / max) | R median |
|---|---|---|---|---|---|---|
| x2 | 0.761 | 0.207 | 0.713 | 0.713 | 0.125 (one matrix) | 0.97 |
| x10 | 0.761 | 0.207 | 0.713 | 0.713 | 0.125 (one matrix) | 0.97 |
| rebuilt | 0.623 | 0.012 | 0.453 | 0.453 | 0.019 / 0.108 / 0.190 | 0.51 |
| new_few | 0.0127 | 0.0127 | 0.0031 | 0.0050 | 0.122 / 0.133 / 0.143 | 0.28 |
| new_thin | 0.0188 | 0.0133 | 0.0134 | 0.0134 | 0.352 (one matrix) | 0.44 |
| random_size | 0.0110 | 0.0018 | 0.0015 | 0.0015 | 0.022 / 0.192 / 1.039 | 0.41 |
| random_kl | 0.0106 | 0.0016 | 0.0014 | 0.0027 | 0.022 / 0.192 / 1.002 | 0.42 |

Stage 0 passed on all four pods: the do-nothing reads exactly 1 with no changed variable, and the identity interchange
gives 0 argmax changes. The pieces plus the remainder equal dW to within 3e-17, well under the filed bar of 1e-12. Every
c is positive. With the needed set held at E268's, the gain and rebuilt edits read the same as with their own set, so
their drop comes from the split, not from a different set.

## The proposed paper line

Proposed: "With the adjoint split (scaled per matrix, as defined), the gain edits read 0.71, rebuilt 0.45 and the new
edits 0.003-0.013, the same ordering as least squares at lower values; without its scale the split does not separate
them."

1. **MUST: the last clause is wrong.** Without its scale, the split still puts the gain edits (0.207) well above the
   new edits (0.013) and the randoms (0.002). What it loses is the rebuilt edit. Rebuilt reads 0.012 there, level with
   the new edits. Suggested: "without its scale, the rebuilt edit reads as low as the new edits (0.012 against
   0.013)."
2. **SHOULD: give the reach.** The scaled split reproduces 97% of the gain edits' weighted energy but only about half of
   the rebuilt edit's (R 0.51). Part of rebuilt's drop from 0.62 to 0.45 may therefore be the split's limit rather
   than the edit. A clause such as "the scaled split reaches about half of the rebuilt edit's change" keeps a reader
   from reading 0.45 as the edit's own value.
3. No verdict wording. The run is descriptive, and the line gives readings only, as P1.1 requires. Keep it that way.

## The filed rivals

- Rival a, as filed: the gain edits "above 0.761 by more than 0.003". Not met: they read 0.713.
- Rival b, as filed in `predictions_run2.json`: the gain edits "at or below their least-squares 0.761, as E329's
  unscaled 0.207 or between". Met as written, because 0.713 lies in that band. The card's version of rival b ("reads
  like the unscaled one ... within 0.05") is not met, because 0.713 is 0.51 above 0.207. The page should report both
  as read. It should not say "the scale does not rescue it": the scale brings back most of the gain edits' share and
  the rebuilt edit's place in the order.

## Not run

- The per-item check of W0 on the model the input statistics are read on (SHOULD 1 of the pre-run check). It was not
  added mid-run, so the result page lists it as not run.
- No independent recompute. This is not a headline answer-paragraph number.
