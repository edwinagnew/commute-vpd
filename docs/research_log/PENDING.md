# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 baseline-lead · reports/analyses/2026-10-30_analysis_e56-keydependent-patch.md, notebooks/E56_keydependent_patch.ipynb, experiments/TRIED.md · E56 card (new: Q/Result, figure b2_curves.png; Run line: done, jobs 46170/46171, $0), claims table (route-split-reads-kind row: five seeds, neuron patching does not reproduce it, one phrase narrowed), timeline · E56: neuron patching (B2) does not read kind on E54's pair: N16 ~0 on both kinds; full-width MLP share separates at matched size (0.94 vs 0.70) but not at the cipher's own step (0.96) and reverses on E12's pair (E52), so it tracks how diffuse the early cipher update is, not kind (Fable's reading, adopted). P1 passed only under the pre-run k50 binning; unbinned (4096 elicit vs 12288 matched cipher on all five seeds) it would have failed. At five seeds r gap 0.52 vs S_lo gap 0.034 under its range, ratio 5.2 (P2); matched cipher r stays a cipher (P3). Next: ask the user for Jain et al.'s readout on the elicit keys
- 2026-10-30 · decisions.md · pending decisions (remove card decision-e55-c2); overview Line/Next · User chose option 2 then 1: a $0 refined, de-confounded rerun of the pragmatic instruments first, then C2 VPD; benchmarking VPD against pragmatic techniques is a priority.
