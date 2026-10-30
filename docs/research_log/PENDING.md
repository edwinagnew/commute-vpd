# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-30 regating-lead: E58 Amendments 2, 3, 4 and 5 (A2-A3 flushed in v148; this line supersedes that text where it differs): reports/analyses/2026-10-30_analysis_e58-open-pair.md §§ "Amendment 2", "Amendment 3", "Amendment 4", "Amendment 5" (opening answer updated) + notebooks/E58_open_pair.ipynb (figures a3_carry, a3_shift, a4_within, a4_shuffle; A5 cell) + experiments/TRIED.md rows "E58 Amendment 2/3/4" (A4 row carries A5) · log places it touches: E58 card, claims table, step-2 status, timeline 2026-10-30 · At 96 trained keys every map carried further and the copy map reads "reuses" on all three seeds (the held-out label's positive control), but over three seeds r differs between the three own-guess maps by more than seed noise (filed within-key-count rule fails at steps 10 and 30; at step 10 not in carry order, rank-1 highest r), so the one-seed "r reads the same" line is dropped; the shuffled elicit map reads r 0.475, cipher_like as filed but near the 0.5 cut, and Amendment 5's seeds 1-2 read 0.485 and 0.471 (mean 0.477, unscored rule: "mostly the key-conditional guess, partly the token kind" stands; no new measurement code, no separate review); Amendment 3's cross-key-count test stays "untested" (cipher r drift 0.14); granularity limit: a new circuit of old subcomponents reads as old.
