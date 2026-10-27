# Lead digest, 2026-10-27 (commute line, E37 Amendment 1)

**Why this session finished:** E37 Amendment 1 ran, was written up and checked, and the log is republished. Its
outcome falls between the branches of the filed decision tree, so the next step is the new Blocking card "A
decomposition trained on a broad fine-tune stays twice as fragile" (`decision-broad-fragile`). Nothing runs on the
commute line until you pick an option there.

**The answer in plain words.** We asked whether training the decomposition to copy the fine-tuned model reaches a
state that is both faithful and robust if it trains four times longer, or starts from the adjoint split. Neither
does. Both end at about twice the original decomposition's adversarial loss, while matching the fine-tuned model's
behaviour. The third arm, which adds VPD's masked losses to the training, was not run. Its control trained with
nothing to copy, and that alone pushed the adversarial check to 3.1 to 3.8 times the original.

## What ran

- **E37 Amendment 1**, plan `reports/plans/2026-10-26_plan_teacher-spread.md` → "Amendment 1 (2026-10-27)".
  - Predictions were filed in e70ee46 before the run. It ran on one H100 for 34 min and cost $2.27.
  - All three controls passed, and the first read reproduced E37's state.
  - P1 held: longer training still fails the check (2.23 at step 4000). P3 held: the adjoint start ends where plain
    training does (1.84 against 1.89 at step 2000). P2 was not scored, because arm 2 never ran.
  - The training loss still fell by 44% over the last doubling, while the check did not fall. The filed tree has no
    branch for that, so sending a card rather than extending the run is a judgement. The card offers the extension.
  - Report: the "Amendment 1" section of `reports/analyses/2026-10-26_analysis_teacher-spread.md`. Notebook
    `notebooks/E37_teacher_spread.ipynb`. Figure `experiments/teacher_spread/figures/amend1_steps.png`.
  - Independent check: `reports/reviews/2026-10-27_review_teacher-spread-amend1-analysis.md`. It found the scores and
    numbers right and asked for the branch wording and the extend option; both fixes are in.
- The research log is republished as v58 with the E37 card, a timeline line and the new card:
  https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu.

## Pending decisions for you

- **`decision-broad-fragile`**, marked Blocking. Its options:
  - a. Extend plain training to 16,000 steps, about $1.5.
  - b. Test whether the frozen Γ is the limit, after a $1 control with the masked losses switched off.
  - c. Train the subcomponents' span. No design exists yet.
  - d. Settle the anchor question. A $1 control moves the decomposition at random by the same small amount.
  - e. Take teacher training to the larger model as it is.
- **`decision-pass-anchor`** is re-rated from No rush to Soon, since option d leans on it.

## Money and problems

- RunPod this session cost $2.43 of the $40 budget: the run, and $0.16 for two failed starts. Earlier sessions today
  spent $6.17, of which $0.34 went on four failed starts.
- remote-kernels `start()` failed six times from 11:13 to 12:02 on the automatic-shutdown watchdog. Restarting
  Claude Code fixed it; the problem record is closed.
- No pods or Slurm jobs are left running. Nothing is in `~/commute-data/blocked.txt`.
- One line stays in `docs/research_log/PENDING.md`: the re-gating line's E38 run, which is not this lead's to flush.
