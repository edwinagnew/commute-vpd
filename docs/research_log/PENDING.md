# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-13 · reports/analyses/2026-11-13_analysis_p3-9-frozen-neuron-share.md · P3.9 cell row (filled), timeline · P3.9 (E179): on the frozen-neuron decomposition p-015e0710 (MLP matrices only) the re-gating share is no longer saturated (sides 0.921-0.995, random changes 0.81-1.06) and 3 of 6 pair cells leave their random band against 0 of 6 on p-9707d6ff; but the gold twin, a near-null pair, is beyond band at both saves with opposite signs, so the one-draw band is too narrow here and no pair is read as differing. Premise check: doubtful. One Vast pod lost before any state ($5.64).
