# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · decisions.md · Pending decisions; decisions · remove card decision-real-vs-twin: the user chose per-word vectors from the training examples and a third trained copy of each fine-tune (P2.4 runs at n=3)
- 2026-11-10 · reports/analyses/2026-11-10_analysis_e141-update-svd.md · table of work (P2.9), method comparison, E141 card, timeline · The effective rank of each saved weight change, from its singular values alone, separates 5 of 9 task pairs against the re-gating share's 7 on the same fine-tunes, agreeing on 3: it misses both pairs of about equal update size and separates both real-vs-twin label pairs the share does not, where the twin trained longer; provisional, labels as built
- 2026-11-10 · decisions.md · Pending decisions; decisions · remove card decision-finetune-first-steps: the user chose option a (overshoot check, rerun 2 of 2 of E139, running)
- 2026-11-10 · reports/plans/2026-11-10_plan_old-task-labels.md · Next; table of work · new cell P1.7 (user): provisional labels of the 17 old 67M tasks under the second wording with E137's search and cut-offs, 2 seeds, 8-token inputs; E142 running on the cluster
- 2026-11-10 · decisions.md · Pending decisions; decisions · remove the card on the share reading about 0.8 on a fine-tune that learned nothing (P2.6, ask 8): parked by the user on ladder rung L2.9
- 2026-11-10 · reports/plans/2026-11-10_plan_built-control-design.md · Next; table of work · P1.8 filled: design of a built control; every built label is in the end linear unreadability, so the control is a contrast of two originals (recommended: the XOR imitated from a 67M first taught a weak XOR against the plain 67M); a run needs a new cell from the user
