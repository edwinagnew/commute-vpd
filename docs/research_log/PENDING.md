# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · docs/research_log/parts/experiments/e12.html · experiments card E12 · put back fig/e12_route_separation.png with its caption (`git show 8109b148 -- docs/research_log/parts/sections/overview.html`): v213 cut it from the overview, and it was in no card; changelog links to #next-regating and #next-commute now dangle, point them at the sections they became.
- 2026-11-04 · reports/analyses/2026-11-03_analysis_e75-rl-reads.md · experiments card E75 · own-samples arm F100, F10 added (job 50874), no lean filed on it: same filed labels as RL and the null but "no call" on in-span against slice at F10 (both reads under the 1e-5 floor), KL(updated ‖ W0) 0.00102 and 0.000294, share sent to Δ 0.435 and 0.406 outside the fine-tunes' range; "queued" in the Moved-on line goes; figures r_vs_n.png and ruler.png regenerated.
