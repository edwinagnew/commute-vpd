# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/analyses/2026-11-05_analysis_e96-locate-mechanism.md, notebooks/E96_locate_mechanism.ipynb, decisions.md · new card e96 (`parts/experiments/e96.html`), timeline, decisions section, claims table · E96 stages 0-2, provisional: a sweep of 24 heads re-finds copying head (2,4) without the prefix score, reads shared; head salience failed and the probe and MLP leans missed; lowercase filed "not found at head or pair grain", about half of its best share is W₀ rising (post hoc), unreadable at this grain with row 121's two behavioural numbers beside; stopped by Fable's advice, no amendment spent.
