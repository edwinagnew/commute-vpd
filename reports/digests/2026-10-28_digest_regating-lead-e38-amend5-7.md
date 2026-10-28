# Regating-lead digest, 2026-10-28: E38 Amendments 5 to 7 done; the 67M restore line waits on your card

The task is done. You chose the direction-matched restore, and it ran with two $0 follow-ups that its checks
recommended. All of it is written up and in the log. Putting back the set Γ switched on takes back about a fifth of the
gain, but a set ranked by pull toward the target takes back as much. The top target-ranked set at the same count takes
back at least what both do together. So Γ finds, without the target, a set rich in target-pointing subcomponents, and
its share is what those members carry. What comes next is your call on one card.

## What ran

All runs are cluster debug jobs on an L40, each under a minute, $0. Each had filed predictions and controls first.
Each also had an independent Opus check, every one of which agreed with the run's verdicts.

- **Amendment 5, the direction-matched restore** (job 40121; you approved it). We ranked the subcomponents outside the
  switched-on set by first-order pull toward the target. We matched them to its count per matrix and put their
  trigger outputs back. Filed tree "outside": they take back more on emoticon (0.35-0.40 vs 0.22-0.24), and the two
  tie on ` which` (about 0.19 each).
- **Amendment 6, both sets together** (job 40196; run as an FYI, $0, recommended by both reports). Together they take
  back about the sum of the two: "separate". That went past the filed "partial", because the plan wrongly assumed
  restores saturate. On emoticon, the switched-on members that point toward the target take back as much as the
  direction-matched set; the rest push against the target.
- **Amendment 7, the settling control** (job 40240; FYI, $0, recommended by the Amendment 6 check). The check showed
  restores add almost linearly, so any two disjoint sets might read "separate". So we ranked the top target-pointing
  set at twice the switched-on count, its members allowed. That set takes back at least what both sets do together on
  every state. The lead is clear on emoticon and within noise on ` which`. Whether "separate" is generic came out
  unscored or split. So the filed tree left the choice to you.
- **Files.** Plans `reports/plans/2026-10-28_plan_e38-amend{5,6,7}.md`. Analyses
  `reports/analyses/2026-10-28_analysis_e38-amend{5,6,7}.md`. Checks
  `reports/reviews/2026-10-28_review_e38-amend{5,6,7}-check.md`. Notebook `notebooks/E38_recruitment_test.ipynb`,
  sections "Amendment 5" to "Amendment 7". `TRIED.md` row "E38 Amendments 2-7". The log, now v75, is
  [Update Decomposition Log](https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu).

## Pending decision

- [`decision-e38-direction`](https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu#decision-e38-direction): a ranking by
  pull toward the target takes back at least what the switched-on set adds. The options:
  - (a) close the 67M restore line and record the reading (recommended in the analysis; the tree does not choose);
  - (b) the upstream step: what switched the set on;
  - (c) carry the question to the Qwen line.
  - Open point, offered as a $0 extra: on ` which`, the switched-on members the ranking drops may take back more than
    first order credits. That restore is not run.

## Deviations, money, problems, friction

- **Deviations.** Amendments 6 and 7 ran without sign-off, under your rule that cheap recommended checks are run (FYI
  in chat each time). The Amendment 5 plan said the switched-on set uses no training windows; it does. A dated
  correction is in the plan.
- **Money.** $0, cluster only, no pods.
- **Problems.** None.
- **Friction** (notes 30-38 in `~/commute-agents/fleet/friction`):
  - Pyright and sacct can't run from the worker or the dev node, so memory requests are guessed.
  - The copied sbatch lacked the FINISHED block.
  - cpu-run can't see the scratchpad.
  - Subagents share one working tree, so uncommitted log edits blocked others' pulls.
  - Two leads' log flushes collided on v73, and the publisher needed four tries.
  - A pod-runner was forced to hand back while its job was still queued, so a fresh watcher had to be started.
- Nothing in `~/commute-data/blocked.txt` from this session. No key or bound was touched.
