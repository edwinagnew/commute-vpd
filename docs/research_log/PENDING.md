# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 baseline-lead · reports/analyses/2026-10-29_analysis_e52-baseline-neurons.md (43d919a, independent check) · e52 card, decision-e52-p4 card · narrow "no small set of neurons carries either" to "the most-shifted neurons up to 256 carry at most 0.24 of the gain"; copy target after 16 examples is about 3e-4 to 4e-4 (1 in 3,000), not 1 in 10,000; "within 0.03 of zero" -> "within 0.031".

- 2026-10-29 qwen-decomp-lead · reports/analyses/2026-10-29_analysis_e44-amend1-layer-cut.md, notebooks/E44_qwen_maps.ipynb (Amendment 1 section), reports/plans/2026-10-28_plan_qwen-q0q1.md (Amendment 1b record), experiments/qwen_maps/results/predictions_amend1b.json · E44 card, pending-decision card on the pass test, timeline · A1 claim narrowed after an independent check: "copying" is the filed upper bound (the splice gives Γ FT's early taps with W0's late ones, so a Γ that reads the change only from consistent inputs would read the same); "option 1 gives up little" becomes "at most little"; do-nothing keeps 93% on A1's 16 rows (1 − K(28)/K(0)), the 97% is E44's 128×512 reading. Amendment 1b filed ($0, not run): Γ reads H(16)/H(24) itself; m_H(16) ≥ 0.5·u(16) → reproduces from consistent inputs (route (b) in substance), ≤ 0.2·u(16) → copying confirmed; lean "reproduces", moderate
- 2026-10-29 baseline-lead · decisions.md (last row) · remove card decision-e52-p4 (user: A then C), decisions section, e52 card next step, timeline · E52 Amendment 1 (prompting check rerun in a fairer format) being written; then C (key-dependent elicit pair) to /design.
- 2026-10-29 baseline-lead · experiments/baseline_neurons/results/predictions_amend1.json, amend1.py, plot_amend1.py, cluster/baseline_neurons_amend1.sbatch, experiments/TRIED.md (E52 amend 1 row) · e52 card, timeline · E52 Amendment 1 filed and coded, not run: P4 re-asked with n = 0/4/16/31 demonstrations next to the query, induction control first (top-1 on a repeated random sequence ≥ 0.5), P4a passes iff copy share ≥ 0.5 at n = 31 and separates copy-higher; 0.3-0.5 partial, < 0.3 the copy label is a mechanism claim only.
