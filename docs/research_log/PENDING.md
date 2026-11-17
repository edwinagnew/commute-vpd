# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · reports/analyses/2026-11-18_analysis_held-model-retention-run1.md · E197 and E201 card / P1.26 status (both runs done) / timeline · The held model passed every code check; one gain per neuron and head keeps almost none of any fine-tune's change, while a per-layer linear map of the original model's units keeps nearly all of it on old-task fine-tunes, favours the reuse member of three of four counted pairs by small but steady margins not ordered by update size, keeps about half of the rebuilt circuit, and keeps little of one attention head grafted onto the frozen model, new by construction (one build, one site); the memorised fact's lower reading is mostly its embedding change (tested, not adopted; no verdict).
