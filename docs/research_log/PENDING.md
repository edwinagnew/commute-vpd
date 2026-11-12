# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · experiments/qwen_pair_library/results/library_page.md · part 3 (Qwen) state of play; experiment card for P3.5 (E160, E164, E165) · Cell P3.5 filled (3/3 runs). The Qwen pair library: at a recipe that learns both sides (1e-5, 300 steps; E165), copy generalises to held-out words (exact match 1.00) while cipher memorises (trained 1.00, held-out 0.01-0.02), on 3/3 seeds; invented and known facts both reach 1.00. But at matched update sizes the new sides are still unlearned, and at step 300 the new sides are about twice the size, so only the gold twin (E164) pairs two learned updates at a matched size. E164 at 2e-6/100 steps learned no new side (cipher 0; lr starved, checked, not a bug). Premise check: doubtful (the "new" label describes the recipe; the part-2 readings, not the library, are what makes no call); for the 11-18 gate. Spend ~$70 (E164) + $37 (E165).
