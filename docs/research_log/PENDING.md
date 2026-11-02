# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · `reports/analyses/2026-11-02_analysis_e62-amend5-head-sweep.md`, check `reports/reviews/2026-11-02_review_e62-amend5-check.md` · card E62 · check applied, filed verdicts unchanged: replace "the 24 falls together come to a quarter of the masked copy" and "no independent check yet"; the falls hold the mask and the lean sits in the mask, so they cannot account for it; effects of a head through the mask's own computation are excluded, so a head-level anchor is not found rather than absent; (2,4) stands out under the mask at 12.7 key sds and 4.8 times the next head; (1,1) and (2,4) show no separation in the target model.
