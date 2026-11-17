# The copy and lookup pair read under the proposed reuse definition, second version

*definition-lead, 2026-11-09. Cell P1.2, due 11-10. No new run: every number comes from a saved file named below. The
wording read under is the second version of `reports/plans/2026-11-09_plan_definition-page.md`, which the user has not
adopted, so each reading is provisional and is not a verdict until a dated `decisions.md` row adopts a wording. The first
version of this re-read, under the head switch-off wording, is in git.*

We re-read the saved copy fine-tunes and their lookup twins under the new wording, with no new run. The copy fine-tunes
are right on values they never trained on and the lookups have no rule on unseen values, but the frozen-body training
that decides whether the copy seeds read "reused" was never run. For `GOALS.md` that means this pair gives no reuse
label yet, and the wording's first real test is the frozen-body run proposed on the page.

## The answer

Under the proposed wording, the three lookup seeds read "new": a table has no answer on an unseen key. The three copy seeds
carry, so they pass the first clause, and the reading stops there: whether the original model's body holds what copying
needs was never measured by frozen-body training. So no copy seed has a verdict yet. Seed 2's head-off share, 0.471, sits
under the old bar of one half; the new wording does not use that number and it is kept beside it, unchanged.

## The reading, seed by seed

| arm, seed | Right on unseen values (bar 0.90) | Frozen body on unseen values | Reading | Beside: gain lost with the copying head off |
|---|---|---|---|---|
| copy, 0 | 0.995 | not run | no verdict, inside step not run | 0.563 |
| copy, 1 | 0.995 | not run | no verdict, inside step not run | 0.545 |
| **copy, 2** | 1.000 | not run | no verdict, inside step not run | **0.471, under the old bar** |
| lookup, 0 | no answer defined for an unseen key | not needed | new | 0.000 |
| lookup, 1 | no answer defined for an unseen key | not needed | new | −0.004 |
| lookup, 2 | no answer defined for an unseen key | not needed | new | −0.003 |

The original model reads 0.250 on the windows read for carrying, so the fine-tunes' scores are learned.

## What this does and does not say

- **Seed 2 is stated, not rescued.** The share is read as filed; the bar was not moved. The new wording simply does not
  use head-off shares in its verdict.
- **The lookup reading follows from the wording's clause 4,** which counts a task with no rule on unseen values as new.
  That is a choice on the page for the user to accept or change.
- **The saved evidence does not reach the wording's inside clause.** The nearest saved thing is the forced copy fine-tune,
  which moved only the two copying heads and reached part of the native gain; it is not the frozen-body fine-tune, and
  it moves weights in a different place.

## How compelling is the evidence

Three seeds per arm, 200 windows read, one marker. Carrying was read on the fine-tunes' saved states by the earlier
definition-table job and has not had an independent check here.

## Where each number is from

- Carrying: `experiments/definition_table/results/stage1.json`, `.states.e62_copy_s<seed>.P1`; the original model's level
  `.w0.e62_copy.P1`.
- Head-off shares: `experiments/induction_cue/results/run.json`, `.states.<arm_seed>.share`.

## Not run

- The frozen-body training on the copy windows, on any seed, for either arm.
- A shuffled-label twin of the copy task: the lookup is the nearest arm on file.
- A read of the forced copy fine-tune under this wording.
- An independent check of this reading.
