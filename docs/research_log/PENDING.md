# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 · reports/decisions/2026-11-26_decision_cluster-worker-stuck.md (ask whip-lead 1) · main, section 3 card decision-cluster-worker-stuck · Update the card: the node now reads "completing*" (not responding), so option 1 is an admin resume of the node. P1.65 went to a pod under the harness cluster-down path, so option 2 is moot. Still blocked: writeup-lead Fig 7 (three matplotlib scripts, no cell for a pod). New option: the user runs them locally, then writeup-lead pushes the held 11-commit batch.
