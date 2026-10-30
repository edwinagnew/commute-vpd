# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 · decisions.md (E55 plan row) · pending decisions (remove card decision-e55-plan), E55 card Next line, timeline · user took the recommendations: Def 2 bounded so g must generalise to held-out values (dated departure from Jain et al. 2024, stated as such); C2 VPD asked again after Stage C's non-VPD scores
- 2026-10-30 · decisions.md (E55 truth-set row) · pending decisions (remove card decision-e55-truth-set), E55 card Next line, timeline · user chose C then A with B: natural-only all-spurious seeds at 1K-2K + identity-prompt fine-tune first, then score the labelled states with an unscored "rule of two counts" class
- 2026-10-30 baseline-lead · decisions.md (last row), reports/plans/2026-10-29_plan_e54-keydependent-elicit.md · decisions section, next section, timeline · E54 approved: a key-dependent map the original model already does weakly (rank 2-20 values), fine-tuned beside a norm-matched cipher; steering vs the route split. Code being written.
