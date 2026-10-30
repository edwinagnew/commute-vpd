# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead · reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md, decisions.md (update-size row) · E55 card (next step), timeline, decisions · plan for E55 C3: pruning revival and reverse fine-tuning ported to the 67M and read on E58's 12 open-pair states, beside r, the stand-in ratio and S, to ask whether the granularity limit is VPD's alone ($0, one cluster debug job; predictions drafted); new standing rule: update size within a recipe cell is a baseline column in every instrument comparison
- 2026-10-30 regating-lead: E58 Amendment 7 · reports/analyses/2026-10-30_analysis_e58-open-pair.md § Amendment 7, reports/analyses/2026-10-30_analysis_regating-synthesis.md, experiments/TRIED.md, figure experiments/open_pair/figures/a7_r.png · E58 card, claims table, re-gating sub-question status, timeline · a new four-word rule over the key embedding's two leading directions, whose words W₀ never gives, carried 0.55-0.64 beyond its null to unseen keys yet read r 0.23-0.25 (cipher band; D vs its shuffled twin -0.034, kind_only), so on the 67M r reads a carrying new rule as new, like a lookup, and does not read carry; r follows whether the answer is W₀'s own guess; granularity limit untested where it bites (a new rule onto answers W₀ already gives); elicit-prior cipher skipped; provisional until the independent check
