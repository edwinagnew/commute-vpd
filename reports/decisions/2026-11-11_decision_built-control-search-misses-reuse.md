# Decision: the search missed the clearest reuse case. What does the definition do now? (rung L1.4)

**Question.** The search missed the clearest reuse case. What does the definition do now?

This is a question about whether the working definition's premise holds, so it is put without a recommendation.

**What happened.** The working wording of reuse, adopted tentatively on 11-10 with cut-offs of 0.5 for $H$ and 0.8 for
$G$, was to be tested on a built control: you said "if it cant detect built, then it will have to be revised". A built
control needs a positive control beside it, a fine-tune the wording should call reuse. Three runs tried today, all on
the free cluster queue.

- **Sum parity, first run.** Stopped at its first check. The plain 67M already reads "the sum of the last two numbers is
  odd" at 0.61 by a linear read, against a bar of 0.55 and 0.50 for shuffled labels. A side reading: yesterday's plain
  exclusive-or fine-tune holds its variable at 0.9997, so a full-strength fine-tune does teach such a variable linearly.
- **Strongly taught original.** Yesterday's plain fine-tune was used as the original and pushed again. The second push
  moved the outputs by only 0.12, under the 0.15 floor: the output was already saturated. Stopped before any fine-tune.
- **Sum parity rerun, bar raised to 0.9 at your request.** The positive control ran first: a fine-tune of the plain 67M
  that acts when the last number is odd. The plain 67M reads that bit exactly, 1.0 at every site. The fine-tune learned
  its target, with its KL divergence from the target 0.04 of that of a half-pushed model. Yet the best $H$ was 0.466 on
  seed 0 and 0.459 on seed 1, under the 0.5 cut-off. By the filed rule the built control is uninformative on this
  layout, and the plain and taught arms did not run.

$H$ is the share of the fine-tune's change in next-token output that is reproduced when an affine map of the original's
activations is written into the original at one site; 0 is none, 1 is all. Near-0.5 readings on clear reuse cases are
not new: the earlier Always on fine-tunes read 0.45 to 0.58, while the Table copies read 0.72. On this layout a low $H$
cannot tell a built skill from a reuse the search missed, so your condition for revising the wording now bears on reuse
as well as on built.

**Options.**

- **(a) A $0 diagnosis on saved states of why $H$ stops near 0.47 on an exactly readable variable.** Runs: reads of the
  saved control fine-tunes and the plain 67M only, no training. It would split the miss into its parts, how much of the
  change the site carries and how much the map's leftover error costs, and try a richer map or another site. Already
  seen: at the input of block 3 the reversion is 0.72 and the map's mismatch 0.04, yet $H$ is 0.47; at the final residual
  the reversion is 0.93 and the mismatch 0.06, yet $H$ is 0.29. It tells us whether the miss is the search's limit or a
  property of what the fine-tune learned. Next: if a richer map or another site reads well above 0.5, the plain arm can
  be run against that control; if not, the choice comes back to (b) or (c). Needs a new cell with 0 or 1 runs, which only
  you open.
- **(b) You revise the wording, or its $H$ cut-off.** Runs: nothing until a revision is written; the change is yours
  alone. It tells us nothing new by itself, but sets what the later runs test. Next: the control and the built arms are
  read again under the revised wording, on saved states where possible.
- **(c) Close L1.4 and write the gate page, due 11-13.** Runs: nothing. The page would say that the wording, as it
  stands, certifies neither reuse nor built on these layouts: the positive control read under the cut-off, and the
  earlier built attempts had no clean control. It tells the paper where the definition stands today. Next: the gate
  decides the definition part with that finding on the page.

**Evidence.** Page `reports/analyses/2026-11-11_analysis_built-control-attempts.md` and its figure
`experiments/built_control_sum/figures/amend1_h_by_site.png`. Results: `experiments/built_control_sum/results/stage0.json`
for the first run; `experiments/built_control_strong/results/stage0.json` and `stage1.json` for the strongly taught
original; `experiments/built_control_sum/results/amend1_stage0.json`, `amend1_stage1.json`, `amend1_control_s0.json`,
`amend1_control_s1.json` and `amend1_readings.json` for the rerun; filed rules
`experiments/built_control_sum/results/predictions_amend1.json`. Plans `reports/plans/2026-11-11_plan_built-control-sum.md`
and `reports/plans/2026-11-11_plan_built-control-strong.md`. The adopted wording and cut-offs: `decisions.md`, 2026-11-10,
"Working definition of reuse".
