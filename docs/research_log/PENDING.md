# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 · reports/analyses/2026-11-23_analysis_p1-58-gain-share.md · cell P1.58 (Part 1 / definition readings), experiments card E282 · Gain-only share prototype (E282, $0): the built new edits read ~0.96 at neurons and at VPD against a bar of 0.3, so the share cannot tell new from reused here; E62 copy vs lookup is no call. Held-out windows reused the fitted cues; the unseen-cue test was not run.
- 2026-11-23 · reports/decisions/2026-11-23_decision_p1-58-gain-share-next.md · pending decisions · New decision card: what to do with P1.58's remaining run after the gain-only share failed its new-edit control (recommended: fit on some cues/keys, read on unseen ones).
