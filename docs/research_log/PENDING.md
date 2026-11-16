# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-16 · definition-lead · timeline, P1.20 row · The ranking test's controls ran again after the search fix: the triggered-neuron control now finds its built pair (rung 2) but needed a third name because the refit stopped short; the user chose to fix the refit and spend the cell's last run on the controls first, then the remaining items.
- 2026-11-16 · compare-lead · timeline, part 2 gate section · Cell P2.26 brought the part 2 gate page up to date (reports/analyses/2026-11-16_analysis_part2-gate-page-update.md). On the 67M the re-gating share alone puts the side built to reuse higher on all nine built pairs, including three where both sides changed the weights by the same amount, but the prior and steering together already give the same nine directions: the share adds size-matched calls and gap sizes, not a new direction. On Qwen it adds nothing yet: saturated on one decomposition, and on the frozen-neuron one the side with the smaller weight change reads the higher share in every pair cell. The user approved two follow-ups the same day: the combined figure drawn on a pod (P2.27) and a size-matched re-read of every reading (P2.28).
