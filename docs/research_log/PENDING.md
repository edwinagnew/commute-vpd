# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 baseline-lead · reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md, notebooks/E54_keydependent_elicit.ipynb, experiments/TRIED.md · E54 card (What it means, Moved on), timeline · Correction after an independent check: drop "about three times"; the route split separates the pair more clearly than steering in every comparison, by ~1.7x (cipher own step) to over 10x (S estimated in log-odds), ~6x over noise floors; both halves of the P2 miss depend on choices (cap miss mostly the nats unit, the vector moves both tasks ~2.5 nats; "no separation" rests on the part-learned matched cipher, P 0.23; at its own step S 0.114/0.066 would separate). Verdict unchanged.
