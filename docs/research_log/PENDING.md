# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 baseline-lead: plan reports/plans/2026-10-29_plan_baselines-scoping.md (baselines to compare VPD's teach-vs-elicit reading against: weight stats, neuron patching, steering, behavioural, SAE); pending-decision card: which baselines to run (user's choice); B1=E51 running at $0.
- 2026-10-29 baseline-lead: E51 (B1, weight-space baselines) done, reports/analyses/2026-10-29_analysis_baseline-weights.md, notebooks/E51_baseline_weights.ipynb: top-10 share and intruder count do not separate copy from cipher; effective rank passes the rule by 3% but tracks update size; new E51 card, baselines pending-decision card (B1 result), timeline.
