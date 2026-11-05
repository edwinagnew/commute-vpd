# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/plans/2026-11-05_plan_e101-carrying-jobs.md · experiments (new E101 card, ongoing), timeline · E101 coded, not run: the essay's neuron reading (top Pile windows, a blind job, a test it can lose) on E98's carrying neurons against block-matched random ones, stage 0 ($0, debug) first; the advice file says E99, the code is E101; option (a), the relay-free neuron share, not run (an attention-only matrix share, not a neuron reading).
- 2026-11-05 · reports/analyses/2026-11-05_analysis_e101-carrying-jobs.md, notebooks/E101_carrying_jobs.ipynb · experiments (E101 card: ran, provisional, no power), claims table, timeline · E101 ran ($0; jobs 52876, 52885; 36 agent calls): the judge tied almost no carrying neuron's Pile job to its task (lowercase 0 of 16, floor 1 of 179), and the positive control, W0's top firers at each task's answer, read 0 of 10 on all three tasks, so the essay-style job reading has no power to tell reuse from new on the 67M; the jobs drive their neurons but near misses fire them too (0.44 vs the essay's 0.11). Next question: a judge shown the task's own windows with task-specific controls, else Qwen3-1.7B.
