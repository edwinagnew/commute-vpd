# Digest: qwen-decomp-lead (2026-11-03)

**Why I finished.** The size test you asked for is done, and what comes next is your decision. The filed verdict stands: a finished maths fine-tune reads above random changes of its size, and a part-way one does not. Three more saved states show the share rising smoothly with size, but the filed rule for them makes no call. Spend was $0.

## Your decision
Is anything left worth running on Qwen to tell a skill the model already had from a new one? The log card is `decision-e61-next`.
- **Stop and write up (recommended).** Nothing runs.
- **Compare the template and invented-facts fine-tunes at equal size.** About 3.5 h, $0. Both are expected to read as random, because the template moves the maths outputs too little, so a null would be the measure's floor rather than an answer.
- **A second maths seed at steps 8 and 100.** About 1.7 h, $0. It changes no reading unless seeds differ by over 0.03.

## What ran
Job 49673, MATS cluster L40, $0, about 77 min, read steps 10, 20 and 50 of the same fine-tune as exploration. Figures: `experiments/qwen_recipe/figures/size_share.png`.

| step | direct share | size | excess over the alive random change |
|---|---|---|---|
| 8 (scored) | 0.0257 | 0.57 | -0.004 |
| 10 | 0.0375 | 0.76 | +0.006 |
| 20 | 0.0653 | 1.06 | +0.029 |
| 50 | 0.0756 | 1.15 | +0.037 |
| 100 (scored) | 0.0769 | 1.17 | +0.038 |

"Follows size" needed step 20 at 0.0669 or above, and "follows training" needed it under 0.06, so the rule reads no call. Size and training rise together along one run, so these reads cannot separate them. Nothing here separates "already there" from "new". Report: `reports/analyses/2026-11-02_analysis_e61-recipe-checks.md`; notebook: `notebooks/E61_qwen_recipe.ipynb`.

## Not checked
- These three reads had no independent check: they are provisional and sit outside the scored verdict, which was independently reproduced.
- Each random change is known at two sizes only, so the excess at the middle sizes rests on an unmeasured straight line.
- The notebook was not re-executed; the stored figure outputs in cells 25 and 28 may lack the three new points. Cell 36 links the current figure.

## Log
Republished as v194. To stay within the main-page length rule the clerk trimmed about 300 words, some of them in other leads' text (re-gating and teacher-training claim rows, Q1 and Q5, Baselines, the teacher-training card, two other decision cards). It reports no number or verdict changed; I saw the removed lines but did not compare each with its replacement. The list is in the v194 changelog row. The old text is in commit `945d5111` if an owner objects; a revert has not been tried.

## Other
Nothing in `blocked.txt` is ours (regating-lead's question is its own). Friction noted: log publish refusal (above), `check.py` quiet flag, foreground `wait-job` re-arms.
