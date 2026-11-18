# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · reports/analyses/2026-11-18_analysis_rho-saved-items.md · E208 card, Part 1 card (working definition), timeline · P1.30 run 2 of 2 (E208): the working definition read as the theory note writes it, on E196's pairs, E195 rebuilt and E206's grafts. Controls hold after amendment 1 (exact zeros counted): turned-up circuit 1, graft in M's slots 0.002, appended grafts 0, E195 rebuilt 0.995. Every fine-tune 0.999+; gaps rank by update size at both seeds (rival B; identity/random-map gap 6.5e-5 and 7.4e-5 below repeat), under today's aggregate, activation patching and a size-aware distance too. Does not separate reuse from new; three definition questions for the user (per-setting distance at a neuron, shared silence, heads weighted as one neuron). Status unchanged: tested, not adopted.
