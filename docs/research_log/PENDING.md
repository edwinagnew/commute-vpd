# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · reports/analyses/2026-11-21_analysis_e233-british-style.md · E233 card (new, part 2), part-2 claims table (share on real text with a twin), timeline 2026-11-20 · on British spelling against a scrambled twin at matched size the re-gating share is low on both sides (British 0.077-0.136, twin 0.124-0.170) and does not put British above the twin, so the filed prediction failed and the card's stop rule applies; effective rank separates them by under one per cent; figure experiments/british_style/figures/real_vs_twin.png
