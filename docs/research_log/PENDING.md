# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 regating-lead · reports/analyses/2026-11-23_analysis_e279-untrained-panel.md · new E279 card (or E272 card addition), next.html P2.66 row · untrained-SVD share on the wide KL-hi panel (user via whip-lead, $0): 9 of 12 tasks read, no order by the tiers seen (tau 0.03, p 0.50; partials 0.10 given base loss, 0.05 given KL; 0.17 with E_b over K), all no call, a moderate one could be missed at n 9; squad 1.27 and summary -1.24 read outside zero to one, so not yet an interpretable share on realistic text; the paper does not cite it as a reuse reading. P2.66 refilled.
