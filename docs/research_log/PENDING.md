# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 reuse-lead · reports/analyses/2026-10-29_analysis_reuse-stage0.md · new card for the reuse line (experiments page) + one timeline line · Stage 0: no repo task has possession set by construction in pretraining; emoticon and ladder (most commute findings) cannot state a non-trivial f_C/X_C; Def 1 vacuous on small token domains without held-out values; Jain PCFG code unreleased, rebuild ~600 lines, ~5-6 H100 h
- 2026-10-29 reuse-lead · reports/plans/2026-10-29_plan_reuse-stage1.md, reports/decisions/2026-10-29_decision_e55-plan.md, reports/reviews/2026-10-29_review_e55-plan.md · new pending-decision card decision-e55-plan; reuse-line card "Running / Due to run" · E55 plan revised after review: ground truth = behaviour check + interchange patch at `=` on the 16 LOW states of Jain Table 6 (pruning, reverse FT, steering scored in C1, VPD in gated C2); Stage B lowercase only; card asks (i) run A now, (ii) whether a lookup g (cipher, copy) counts as reuse under Def 2, (iii) C2 go-ahead after rehearsal; Due to run: Stage A on the cluster after code review
