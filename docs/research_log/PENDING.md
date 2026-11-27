# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 · reports/analyses/2026-11-26_analysis_p1-65-readthrough-fills.md · experiments (P1.65 card: E321, E322), timeline · P1.65 filled: graded-retained score over the necessary set still 0.999 or more on all 20 fine-tunes (min 0.99936); the rescale-only control at three seeds reads near the full fine-tunes at VPD (higher on copy, about half on cipher) and two to six times above them at SAE latents on every task; the SAE error share is higher for the full fine-tunes (0.68-0.76) than the control (0.51-0.69). Redteam passed both runs.
