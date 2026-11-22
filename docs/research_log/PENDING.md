# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 compare-lead · reports/analyses/2026-11-21_analysis_p2-55-khead-cued-copy.md · E252 card (new), P2.54 2x2 card, claims row (claim 2), timeline · E252 k-head label: with the top 3 copy heads off (2,4/2,3/2,2; (1,1) excluded) key copy loses 0.52, low copy 0.49, ciphers 0.02-0.05; random 3 heads cost 0.08-0.16 (filed floor 0.1 missed); the (2,4)+(1,1) pair costs the ciphers 0.12-0.13, so (1,1) is a general head; K2, K5-K7 hold, K1, K3, K4 miss
- 2026-11-21 compare-lead · reports/analyses/2026-11-21_analysis_p2-55-khead-cued-copy.md · E253 card (new), P2.54 2x2 card (figure replaced by experiments/cued_copy/figures/prior_mech_2x2_cued.png), claims row (claim 2), next (P2.55 filled), timeline · E253 cued copy, a copy-map task at the cipher prior (log10 -5.86, by keeping its own lowest-prior rows): r 0.92 against the cipher 0.27, mechanism effect 0.52 vs prior -0.025; pooled steering, effective rank and top-64 neurons also follow mechanism, so r is not yet told apart from them; answer-absent check missed (-0.33 to -0.35 of the gain); one task family, 3 seeds, provisional
