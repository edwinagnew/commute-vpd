# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 baseline-lead · reports/digests/2026-11-02_digest_baseline-lead.md, reports/analyses/2026-11-02_analysis_e66-interchange.md, reports/plans/2026-11-02_plan_e65-harder-counting.md, reports/plans/2026-11-02_advice_e66-next-FABLE.md · card decision-benchmark-next (new, pending, labelled baseline-lead), section 3's count · nothing is running on the benchmark line and it waits on the user. New card, cold-reader form, question: does VPD against cheaper methods get another attempt, now that the context patch gave no 67M fine-tune a causal label? Options: (a) leave it: the table's rows stay by assumption and the log says the comparison is unanswered on the 67M; no compute. (b) Harder counting test: about 3.5 to 6 hours of free jobs, ranks the cheap methods on a causal truth, VPD can only tie or lose; its saved states sit in the reuse lead's clone or need about an hour's rebuild. (c) Plan a patch that swaps the key on the class map, the one row where VPD could beat every cheap method: needs a new plan and retraining, Fable warns it may inherit the off-target drift seen here. Recommendation: (a), since (b) can't show VPD winning and (c) has no working control yet.
