# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 compare-lead · P2.21 filled (L2.2, E158 job 58971, $0): neuron readings on the case and repeat label twins; case no call, repeat twin higher on every seed (reverse of part 1). P2.20 page and figure updated (reports/analyses/2026-11-11_analysis_p2-20-part1-vs-methods.md); method table final.json tally twin neuron columns now no call 3, apart-not-share 1. Update the P2.20 row/E133 Found bullet so no neuron reading is listed missing.
