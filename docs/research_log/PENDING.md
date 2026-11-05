# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 mechanism-detection-lead: E97 locator, first run (jobs 52618, 52646, $0): plants exact; removal re-finds (2,4) on copy in the state's own terms (0.56-0.65, lookup spared), carry ranks mlp3 first so stage 3 held; lowercase pair (1,1)+(1,2) meets the own-terms rule (0.80, 0.85) but W₀ falls more (1.10 nats), 0.10-0.13 in the gap unit: a shared part, not a removal of the gain. Amendment 1 (edges, EAP-IG form, user's suggestion) coded next. New E97 card.
