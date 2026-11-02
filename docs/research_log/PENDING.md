# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 baseline-lead · reports/plans/2026-11-02_plan_e65-harder-counting.md, reports/plans/2026-11-02_advice_e65-review-FABLE.md · card decision-e65-or-stage-b (new, pending), next · the plan for the harder counting test is drafted and reviewed, not approved. New card, cold-reader form: run E65 with Fable's changes (ranks cheap methods on a causal truth anchored at the unchanged model; VPD cannot win it; $0, 3.5-6 h debug) or plan the interchange patch on the 67M first (gives the 67M rows a causal label, where VPD already has readings, so VPD can be compared with cheaper methods). Recommendation: the patch first. Fable's check also confirmed that Stage A's patch reads 0 by construction on new-count states, so 'new' there rests on behaviour; no label moves.
