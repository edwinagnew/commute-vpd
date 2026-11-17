# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).


- 2026-11-17 · parts/sections/decisions.html · decisions · remove the card decision-p1-24-next-cell (ask 43 answered in chat: none of options (a)-(d); new cell P1.25 uses the definition draft's fixed choices: neuron level, 1 - cos^2, re-pairing within a layer; run 1 code checks and a fine-tuned rebuilt item, run 2 the five trained pairs). Section next: P1.24 row's "Next cell: your card" becomes "Next cell: P1.25"; add row P1.25 as open, due 11-18, 0 of 2 runs.
