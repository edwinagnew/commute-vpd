# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 · reports/analyses/2026-11-24_analysis_p1-59-why-readings-fail.md · P1.59 card (new), E285 card · Why the needed-set readings fail on trained fine-tunes: the filed top-1%-by-flips kept share failed its control; an exploratory change-weighted reading on the largest changes reads 0.76/1 on doubling edits, 0.016-0.041 on every trained class (toy, procedural, RL, realistic alike), 0.011 or less on randoms; arithmetic reproduced, scans not redone; redteam verdict pending
- 2026-11-24 · reports/analyses/2026-11-24_analysis_p2-69-eapig-overlap.md · P2.69 card, E287 card, baselines in the claims table · Wang et al.'s EAP-IG circuit overlap, their pinned code on our items: both built controls pass (doubling edit keeps all 37 edges; built new edits read 0.50 under M's resampling range), and at matched output change a random update reads 0.82; E62 and four key-grid tasks give no headline (circuits under Wang's 70% faithfulness); key-grid copy reads 0.29-0.30 ("new"), but M's KL gap there is 8x smaller than the fine-tune's, so it compares against a circuit for a much weaker behaviour; reuse of M's heads is not tested (revised after redteam review)
