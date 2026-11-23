# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 regating-lead · reports/analyses/2026-11-23_analysis_convergence-panel.md · E272 card, next.html P2.66 row · post-hoc checks (user-approved via whip-lead, $0, redteam-reviewed): seed agreement does not explain the E_b reversal (-0.69 -> -0.67 held fixed; partly step mismatch on arXiv/summary); steps to learn too coarse to read (5/12 tie at step 1), no reading follows it at p < 0.05; keep the re-gating 0.49 (p 0.066) out of the card. P2.66 refilled.
