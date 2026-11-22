# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 compare-lead · reports/analyses/2026-11-21_analysis_p2-51-proxies-vs-r.md · P2.51 card, claims row (claim 2) · proxies rescored against E_b (definition (b)) from E257's saved items: on all 94 trained fine-tunes E_b reads 0.015-0.062, 1.4-5.7x a random update's 0.011, no pair gap favours reuse beyond it, so rank agreement on pairs and key tasks is uninformative; on the 5 built edits (E_b 0.012-0.76) the best proxy, the log-odds share, agrees at 0.20 (0.90 under wording A); pair gaps follow update size at 0.77 (0.80 under A); figure experiments/proxies_vs_r/figures/gaps_vs_r_b_and_r_a.png; RL states pending
