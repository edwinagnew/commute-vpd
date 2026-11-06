# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/plans/2026-11-06_advice_neuron-baseline-next.md · pending decisions · New card (baseline-lead owns): after E102 stage B closed the neuron route on Qwen, run a same-format known-answer pair (TOFU real-person facts vs anchor 2's fictional authors, same recipe) now, lifting the default 'no new example update before 11-11', or write up the comparator table as it stands; Fable recommends the pair; waits on the user.
- 2026-11-06 · reports/analyses/2026-11-05_analysis_e102-qwen-carrying.md (section "Stage B"), notebooks/E102_qwen_carrying.ipynb · E102 card, claims table, method table, timeline · E102 stage B ran (job 53068, $0, provisional): agents wrote jobs for 770 Qwen carrying, control and random neurons and a forced-choice judge tied each to one of four tasks; the power gate read no call (W0 biography picks tied to author facts 0.50 of 90 against random 0.36 of 210, gap 0.14 under the filed 0.2, above chance 0.25), so the layout and author-facts predictions are not read and the neuron route is closed on Qwen by its filed gate; explained share 0.65-0.74 against E101's 0.82 on the 67M. A no call opens no amendment.
