# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · experiments/TRIED.md · experiments card E78 · new row, not run yet: anchor 1 retrained at E61's rate, δW in fp32 at steps 1, 2, 3, 5, 10, 20, 50, 100 (`reports/plans/2026-11-03_plan_e78-anchor1.md`); the card waits for the run.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e69-grpo-loop.md · experiments card E69, claims table, sub-question status, timeline · report and `notebooks/E69_qwen_rl.ipynb`, figures `experiments/qwen_rl/figures/`, TRIED row rewritten: claim as corrected after the independent check (`reports/reviews/2026-11-03_review_e69-independent-check.md`, no bug): under the strict grader RL beats this shuffled-reward run on the same problems by three to eight points (+3.2, +3.1 on answers all three states finish; se near 1 over problems at fixed weights); with format ignored on those answers +1.2, +0.7, not separable from zero; accepting forms like "So the answer is 20" takes the strict lead to +1.25, +0.81, so it may be largely answer format and better maths is not shown; each arm one seed, spread between runs unmeasured; the null gained +5.6 (filed < 1.5), cause open; P7's first part reversed (SFT's KL 113x smaller than RL's, filed 3x larger).
- 2026-11-03 · reports/plans/2026-11-03_plan_e69-amend1.md · experiments card E69, timeline · adopted, not run: second seed of RL and the null on strict and format-relaxed graders, greedy accuracy at saved steps, forced openings (~$20).
