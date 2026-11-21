# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · sae-lead · E245 card (P1.40 run 2, built edits): R under wording A at SAE latents (attn/mlp SAEs, relative bar 0.1 KL(F||M)): reuse x2 0.999, x10 0.85, rebuilt 0.69, moved not read; new new_thin 0.57, new_few 0.52; order holds, gap 0.12 above split-half band 0.11 (rival a, narrowly); every search capped, so lower bounds; remainder share high on x2 too (0.89), so it does not separate new from reuse. Keys next. summary experiments/sae_r_a/results/summary_built.json.
