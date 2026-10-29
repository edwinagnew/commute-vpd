# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · reports/plans/2026-10-29_plan_regating-elicitation-DRAFT.md · pending decisions (new card decision-regating-elicitation) · Fable's advice on developing the re-gating line around 'can VPD tell eliciting an existing mechanism from building a new one': options A ($0, E38's restore share in the real fine-tuned model on E12's copy vs random-map updates), B (carry the separating number to Qwen stage 1), C (cue-free 67M pair); recommendation A then B; awaits the user.
