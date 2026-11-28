# Check of E330, the gain-matched random references for the Qwen re-gating share (review, 2026-11-27)

I re-read every draw E330 saved and checked regating-lead's ranges and their three questions. The ranges are right.
With these references, the fine-tunes' share sits above what a random change of the same answer gain gets on the new
tasks and on the maths tasks. It does not sit above on the two reuse tasks, copy and known, for either the
decomposition or the neuron control. This closes MUST 1 of the P2.90 review, provided every comparative sentence is
written per task with the number of draws. The random shares are read at a gain of about 1. That is the floor the code
itself documents as unstable, so they are far noisier than the fine-tunes' shares, and each draw should be shown.

Sources: `experiments/share_unified/qwen_random.py`, `results/qwen_random_{vpd0,vpd1,neuron0,neuron1}.jsonl` at
df4c220d; fine-tune shares from E328 as checked in `2026-11-27_review_p290-share-unified.md`.

## Every draw (share = numerator / gain; "cap" = never reached the gain band, share not read)

| Task | VPD draws 0 / 1 / 2 | VPD fine-tunes | Neuron draws 0 / 1 / 2 | Neuron fine-tunes |
|---|---|---|---|---|
| copy | 0.719 / 0.630 / 0.666 | 0.61-0.64 | 0.434 / 0.534 / 0.462 | 0.42-0.46 |
| known | 0.687 / 0.515 / 1.031 | 0.58-0.60 | 0.373 / 0.229 / 0.726 | 0.30-0.31 |
| cipher | cap / 0.375 / 0.351 | 0.56-0.58 | cap / cap / 0.086 | 0.30 |
| invented | cap / 0.397 / 0.375 | 0.64-0.65 | cap / cap / −0.277 | 0.35 |
| gold | 0.554 / 0.553 / 0.483 | 0.74 | cap / −0.0004 / 0.055 | 0.30 |
| gold_shuf | 0.551 / 0.538 / 0.478 | 0.71 | cap / −0.017 / 0.051 | 0.30 |

Read gains lie between 1.010 and 1.093, inside the filed band. The zero change reads gain and numerator exactly 0 on
every row set. That is a determinism check, as in MUST 2 of the earlier review. Eight of 36 draws hit the cap. Their
largest gains were 0.0002 to 0.84.

## The three questions

**(a) A share of 1.03 on VPD known draw 2.** The share is a ratio of two signed means, the re-gated model's rise over
the full model's rise, and nothing bounds it at 1. Here the numerator is 1.070 and the gain 1.037. The denominator is
not broken: the gain clears the floor of 1. But `local_faith.MIN_GAIN` says shares "are unstable" below a gain of 1,
and every random draw sits within 10% of that floor. So 1.03 is a valid reading but not a stable one. It should be
shown as one draw, not folded into a range that reads as a measured interval.

**(b) Comparing shares at a gain of about 1 with fine-tunes at 10 to 160.** The comparison is the one the 67M paper
makes (P2.82, E310), so it is consistent across the paper. It is not a like-for-like match, though. Two things differ:

- Size. The random changes are 0.8 to 31 times the fine-tune's per-matrix norms, yet they move the answers by only one
  log-odds.
- Noise. At a gain of 1, any fixed error in the numerator is 10 to 160 times larger as a share than at the fine-tunes'
  gains.

The text should say this once, in plain words: the reference is a random change scaled until it moves the trained
answers by one log-odds, much less than the fine-tunes move them. A random change matched to the fine-tunes' own gain
was not reachable: the cap at 100 times gave gains of −7 on VPD copy.

**(c) Draw-to-draw spread.** The ranges below are the largest minus the smallest draw. The fine-tunes' spread across
seeds is 0.03 or less on every side.

| Task | VPD spread | Neuron spread |
|---|---|---|
| copy | 0.09 | 0.10 |
| known | 0.52 | 0.50 |
| cipher | 0.02 (two draws) | one draw only |
| invented | 0.02 (two draws) | one draw only |
| gold | 0.07 | 0.06 (two draws) |
| gold_shuf | 0.07 | 0.07 (two draws) |

The known side's spread is as wide as the gap between any two readings in the paper. One known draw alone can land
above or below the fine-tunes.

## What may and may not be said

- MUST: give comparative sentences per task, with the number of draws read. Allowed: on VPD, the fine-tunes' share is
  above every gain-matched random draw on cipher (0.56-0.58 against 0.35-0.38, two draws), invented (0.64-0.65 against
  0.38-0.40, two draws), gold (0.74 against 0.48-0.55) and gold_shuf (0.71 against 0.48-0.55). On copy (0.61-0.64
  against 0.63-0.72) and known (0.58-0.60 against 0.51-1.03), a random change of the same answer gain reads as high as
  the fine-tunes. The neuron control shows the same pattern: above on gold and gold_shuf (0.30 against 0.06 or less),
  overlapping on copy and known. On cipher and invented the neuron comparison rests on one draw each. It must say so,
  or be left out.
- MUST: no sentence may say the Qwen fine-tunes are "re-gated more than a random change" without naming the task. The
  reuse tasks are exactly where it fails. This finding bears on claim 2, so whip-lead must see it before any paper
  wording.
- SHOULD: show every draw (dots), not min-max ranges, because of the known spread. Name the capped draws in the caption
  or the appendix, not only in the CSV. Leaving them out is not neutral: these are the draws whose direction could not
  raise the answers, and they cluster on the new tasks.
- No "works" or "fails" wording: the definition is still under revision (P1.1).

## Not run

- No measurement of the numerator's noise at a gain of 1, such as two read seeds on one draw. The zero control gives
  exactly 0 and says nothing about it. One repeat read of VPD known draw 2 with fresh rows would show whether its 1.03
  is the draw or the read.
- No check of the KL of the randoms against the fine-tunes'. It was reported, not matched (0.0003 to 0.31 here).
- No independent recompute: these numbers are not in the answer paragraph.
