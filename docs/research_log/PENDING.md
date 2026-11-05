# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 qwen-decomp-lead · decisions.md, reports/plans/2026-11-05_plan_qwen-reorient-amend1.md · pending decisions (remove card decision-after-e76), decisions section (one line), D1 status line · Ask 31 decided (user): option A without the pod. D1 is judged by the clean-model table, share left in Δ, squared size and behaviour kept under own masks; the adversary is a stated limit in the write-up, with E89's do-nothing fail on the larger TOFU update beside it; no further adversary runs. Ask 32: E94 (D3 rows on both anchors) is being coded, stage 0 first.
- 2026-11-05 definition-lead · reports/reviews/2026-11-05_review_e95-independent-check.md · E95 card, E90 card (copy rows) · E95 independent check: holds with caveat. The key reader loses .74-.77 against .50-.53 for the copy states on the same windows and scoring, in every unit; the copy maps get no label. Caveat: the 24-key copy map and the replicate copy map have no reader on their own table, so their 'needs the input token' is carried over from the first table, not measured.
