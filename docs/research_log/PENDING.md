# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 · reports/analyses/2026-10-30_analysis_e55-amend5.md, reports/reviews/2026-10-30_review_e55-amend5-analysis.md · E55 card, timeline, section 3 · E55 Amendment 5 checked and corrected: unseen (not fresh) rows, P4 met but untestable, 16 reuses states (17 with W1s1) against 9 new; which saves enter Stage C's truth set is fixed by the Stage C plan; fast copy is a paired comparison, not an independent run; six departures added; cards decision-e55-plan and decision-e55-truth-set removed with the user's decisions recorded in the E55 Next line and timeline; job IDs 46022 and 46023.
- 2026-10-30 · baseline-lead · reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md (section "Amendment 1"), notebooks/E54_keydependent_elicit.ipynb, experiments/keydependent_elicit/figures/a1_units.png, experiments/TRIED.md · E54 card · E54 Amendment 1 (job 46070, $0): steering's share read in log-odds, the re-gating share's unit, still does not separate the elicit map from the matched cipher (0.25, 0.25 vs 0.22, 0.18; gap 0.05 under the 0.2 minimum); the re-gating share's gap is 10.3 times steering's, measured, replacing the estimate "over tenfold"; headline unchanged; in log-odds the elicit steering share is under the 0.3 cap the nats share missed (P2 stays failed as filed).
