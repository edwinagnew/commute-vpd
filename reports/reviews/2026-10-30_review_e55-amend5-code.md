# E55 Amendment 5: pre-run code review, record (2026-10-30)

Scope: `experiments/pcfg_anchors/amend5.py`, `experiments/pcfg_anchors/results/predictions_amend5.json` and the
`amend5` stage of `cluster/pcfg_anchors.sbatch`, as filed at 30ab4f6. Reviewer: a fresh agent that wrote neither the
code nor the plan. The fixes were committed at 7b5ef02, before either job ran (46022, 46023). This file records the
review after the fact; the analysis is `reports/analyses/2026-10-30_analysis_e55-amend5.md`.

## Before the review: the code writer's note on the prompt control

The plan's prompt control asked for the count of a under `C c 40`. The code writer pointed out that LOW answers the
count of a on `C b 40` 0.9999 of the time, so on `C c 40` it almost surely already does: the arm would be a
do-nothing control. The lead switched it to `T a 40`, a pretrained prompt that asked for a terminal and never took a
letter, and added a W₀ control that stops the run unless LOW's behaviour share for the count of a there is at most
0.05 (plan Revision 1, commit 8a83aec). The run confirmed the note: LOW at `C c 40` reads 1.000, and at `T a 40` 0.000.

## Findings and what was done

| # | finding | done |
|---|---|---|
| 1 | The shortcut share was unconditioned: the share of all training-distribution rows answered 14, bar 0.6. A correct answerer scores 0.69, since 69% of rows are labelled 14, so any state that changed route would have been classed a shortcut. | Fixed: conditioned on the label, the share of rows labelled other than 14 that are answered 14; a correct answerer reads 0, a constant one 1. Bar 0.5. Dated amendment in the predictions file. |
| 2 | The shortcut prediction had no scored scope: no step and no hit or miss rule. | Fixed: scored at 1K, hit iff no natural-row state there is classed shortcut; 500 and 2K described. |
| 3 | The evaluation streams were called fresh, but they are the streams Amendment 2 drew its rows from. | Wording fixed in the predictions; the code's docstring and comment still say "fresh", to be fixed with the next code commit. |
| 4 | The prompt control's W₀ check also stops the natural-row arms, which do not depend on it. | Accepted: a miss there means the setup is wrong, and the arms are cheap to rerun. |
| 5 | The `void_if` wording was ambiguous about which control states void the natural-row predictions. | Fixed: void iff none of the six control states, two seeds by three saves, reads count a through. |
| 6 | Small inconsistencies between the docstring, the predictions and the plan; the fast-copy lean still listed the unreachable "both" class. | Fixed. |
| 7 | The two sbatch jobs, controls and arms, write the same `amend5.json`. | A comment was added to the sbatch file; the arms job runs after the controls job finishes. |

No finding changed a measurement after any model was read. The predictions file's `amendments` list records the
filing changes with their date.
