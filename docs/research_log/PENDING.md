# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/plans/2026-11-06_plan_e103-known-answer-pair.md, experiments/TRIED.md (E103 row) · experiments page (new E103 card, coded not run), What's next (neuron baseline line) · E103, the known-answer pair, coded: Qwen fine-tuned on TOFU's 217 real-fact rows with E84's recipe beside the fictional-author fine-tune, so only the label differs; control first (gain at matched size at least half the template's), then E102's neuron readings on both, and E94's VPD reads asked of qwen-decomp-lead.
