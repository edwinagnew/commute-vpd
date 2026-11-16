# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-16 compare-lead · P2.28 / E187 size-matched readings: reports/analyses/2026-11-16_analysis_p2-28-size-matched.md — on the 67M, shrinking the larger update to its partner's size leaves the share separating 9 of 9 built pairs (steering 7, top neurons 8, neurons for half the gain 5); on Qwen the share separates 1 of 6 pair-and-save cells (was 3), and the top-neuron overlap reverses on copy and cipher. Premise check doubtful (a rescaled update is not a smaller learned one). Descriptive, no verdict. Touches: P2.28 card (filled), part 2 claims table (size), gate-page summary, timeline 11-16.
