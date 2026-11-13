# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · reports/analyses/2026-11-12_analysis_third-wording-no-reading.md · Part 1 definition (third wording, testing only) · P1.18 filled: the user's third wording (reuse as a soft intervention on a variable of the original), not adopted, read by argument on every item: it reads built edits as built where the variable was fixed before the edit, gives Always on and Table no reading as written, and leaves the memorised fact and the old new sides to the user's choice of map class.
- 2026-11-13 · reports/analyses/2026-11-13_analysis_third-wording-measured.md · Part 1 definition (third wording, testing only); experiments card E173 · P1.19 filled (3 of 4 runs, $0): every built control read its constructed grade at all bars; every fine-tune (memorised fact, twin, eight old matched pairs, two seeds) read "no carrying variable among the named ones" (carry at most 0.33); twin against fact no call; the reuse side has the higher trigger share in every pair (descriptive). Table control measured only (user, 2026-11-12).
