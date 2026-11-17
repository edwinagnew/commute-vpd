# E139 amendment: overshoot check replaces the first-steps bar (2026-11-10)

Cell P1.4, experiment E139, ask 23. The user chose option a of the pending-decision card `decision-finetune-first-steps`
(task file, 2026-11-10 10:18). `results/predictions.json` is unchanged; this report amends one early check only.

**What changed.** The filed check stopped a fine-tune when, at step 10, its divergence from the original model M
exceeded 0.25 of KL(F || M), where F is the pair being imitated. That share was a judgement, never measured, and it
stopped every item while the distance to F fell smoothly. It is replaced by an overshoot check aimed at what it was for:
at step 10 the job stops unless KL(F || fine-tune) < KL(F || M) **and** KL(M || fine-tune) <= KL(F || M).
`experiments/reuse_variable_finetune/run.py`, function `train`. The filed `first_steps_share` threshold is no longer read.

**Chosen after seeing the curves.** The stopped runs, moved to `results/stopped_first_steps_bar/`, pass it:

| item | KL(F‖M) | KL from M, step 10 | KL to F, step 10 |
|---|---|---|---|
| Always on, seeds 0/1/2 | 0.139 | 0.104 / 0.107 / 0.103 | 0.036 / 0.037 / 0.036 |
| XOR, seed 0 | 0.078 | 0.041 | 0.041 |

**Unchanged.** Recipe, seeds, the XOR-learned check, the gap floor, the cut-offs for G and H, the label rule, and the
control (passed, job 57251, not rerun). This is rerun 2 of 2 of E139; another failure closes P1.4 with no reading.
