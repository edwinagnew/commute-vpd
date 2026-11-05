# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/plans/2026-11-05_plan_e101-carrying-jobs.md · experiments (new E101 card, ongoing), timeline · E101 coded, not run: the essay's neuron reading (top Pile windows, a blind job, a test it can lose) on E98's carrying neurons against block-matched random ones, stage 0 ($0, debug) first; the advice file says E99, the code is E101; option (a), the relay-free neuron share, not run (an attention-only matrix share, not a neuron reading).
