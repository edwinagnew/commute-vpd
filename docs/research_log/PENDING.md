# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 baseline-lead · experiments/keydependent_elicit/results/{stage0,stage1}.json, predictions.json (amended_2026-10-30_floor), decisions.md · E54 card (Running line) · Job 45963: stage 0 controls all matched; stage 1 stopped on the held-out elicit rate (geometric mean 1.7e-3 below 5e-3, an averaging gap); user lowered the floor to 1e-3; rerunning.
- 2026-10-30 regating-lead · reports/analyses/2026-10-29_analysis_e53-performed-anchor.md (section "Amendment 1b"), notebooks/E53_performed_anchor.ipynb, experiments/performed_anchor/figures/a1b_{breadth,ratio,gain}.png, experiments/TRIED.md · E53 card (Amendment 1b result), claims table (E53 kind-vs-breadth row), timeline · E53 Amendment 1b (jobs 45971, 45972): not read (swap missed): the 500-word cipher's breadth 7.8 vs lowercase 4.5-4.9, twice as broad as E53's cipher (15) but still narrower; E53's confound holds on the true off-target KL; provisional, unscored: its step-300 ratio 0.27-0.29 ≈ E53 cipher 0.26 (0.47 at matched gain); random δW null uninformative (subcomponent gain 0.009).
