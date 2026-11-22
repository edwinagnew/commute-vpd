# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 · reports/analyses/2026-11-22_analysis_reading-admissibility.md, theory/2026-11-22_definition_frozen.md · overview definition card (id existing-new), glossary (reuse, write share), changelog · P1.53 (definition-lead): the user FROZE the definition (sentence and exact form, theory/2026-11-22_definition_frozen.md sections 1-2): the definition card now states the frozen sentence and exact form (kept = one positive gain on the parent settings that occur in F's runs on D; needed = single-variable interchange; R = kept among needed; full reuse iff R = K); the write share E_b becomes a reading, not the definition; P1.53 code check: neither saved reading is admissible as computed (squared cosine: settings from both runs, per-setting gain, sign-blind; E_b: right settings and exact top on kept variables, but scored over K* from mean ablation, not the needed set K).
