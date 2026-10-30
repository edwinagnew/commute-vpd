# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-30 qwen-decomp-lead: E57 card (Blocking for Stage D): row 1 has no teeth at scale, both random controls read re-gating share ~0.99; options A-D and Fable advice in reports/decisions/2026-10-30_decision_e57-no-teeth.md; E57 experiment card: B2 and C0 done, C running.
- 2026-10-30 regating-lead: E58 Amendments 2 and 3 written up: reports/analyses/2026-10-30_analysis_e58-open-pair.md §§ "Amendment 2", "Amendment 3" (opening answer updated) + notebooks/E58_open_pair.ipynb (new section, figures a3_carry.png, a3_shift.png) + experiments/TRIED.md rows "E58 Amendment 2", "E58 Amendment 3" · log places it touches: E58 card, claims table, step-2 status, timeline 2026-10-30 · At 96 trained keys every map carried further (elicit 0.21 then 0.10, copy 0.82 "reuses", the label's first positive control on one seed) while r and the stand-in ratio stayed within tolerance on all three maps, but the no-rule cipher's r rose 0.14 with the key count, so the filed test reads "untested" (Amendment 2 had stopped on its mis-specified elicit control, x 0.213, a miss that stands); granularity limit: a new circuit of old subcomponents reads as old.
