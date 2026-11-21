# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-20 · decisions.md · definition box (readings), decisions · Measurement change, not wording: R under wording A (P1.46/E242) reads 'behaviour preserved' with a relative bar, KL(F || F_S) <= 0.1 KL(F || M); E226's absolute allowance let the do-nothing, random controls and new_few read R = 1.
- 2026-11-20 · sae-lead · E234 card (P1.40 run 1): base-only SAEs at all 13 sites of the 67M; base FVE 0.875-0.995, recovered 0.926-0.999, dead 0.53-0.62 at attn0-3 (a miss); 46 of 75 states drop FVE > 0.02 somewhere; x10 resid_post3 outlier (drop 16) under check. TRIED.md row E234.
