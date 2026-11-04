# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/analyses/2026-11-04_analysis_e85-adversary-teeth.md, notebooks/E85_qwen_d1_teeth.ipynb, experiments/TRIED.md · experiments card E85: link report and notebook; decision-after-e76 evidence · E85 written up, provisional, independent check running: filed outcome "the adversary cannot fail a bad state here" (random 1.009, rarely-on 1.270; bars 1.40, 1.61); beside the rule, rarely-on sits on Qwen's own pass bar near 1.27 and its mean ratio is 1.62 at step 10 of the search; random below do-nothing, no cause claimed.
- 2026-11-04 · reports/analyses/2026-11-04_analysis_e86-update-fractions.md, notebooks/E86_qwen_d1_fractions.ipynb, experiments/TRIED.md · experiments card E86: link report and notebook; decision-after-e76 evidence · E86 written up, provisional: kept at a tenth of the update u1 0.93 saturated, gold 0.83 builds partly, TOFU 0.28 builds graded; clean table works on all three, least squares leaves 0.090, 0.088, 0.098 in Δ; no rule in the table fails wrong columns.
- 2026-11-04 · reports/analyses/2026-11-04_analysis_e82-teacher-on-e68.md, notebooks/E82_qwen_pass_test.ipynb · experiments card E82: link report and notebook · E82's notebook, a thin explanation of the report: teacher training closes 0.083 of the masked KL gap against the whole update's 0.099; no new numbers.
