# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · experiments/recruitment_test/ (a530663) · decisions (card decision-regating-reading, "Running / Due to run" line) · E38 recruitment test running: stage 0 (apparatus) on the cluster, stages 1-3 after it passes; code under independent review.
- 2026-10-27 · reports/analyses/2026-10-27_analysis_recruitment-test.md, notebooks/E38_recruitment_test.ipynb, experiments/TRIED.md · experiments (new card e38), decisions (decision-regating-reading: "What we know" bullets, Running line removed), Q5 status, timeline · E38 recruitment test: removing the recruited set from the fine-tuned weights takes a third to a half of the gain on two of three tasks, far above same-size random sets, but removing it from the original model already takes a sixth to a quarter (P3, lean high, reversed on all six states); stage 1 "outside the tree"; P4 holds on " which", misses on " we"; the census had no power. Independent check: no code artefact; missing controls named.
- 2026-10-27 · reports/analyses/2026-10-27_analysis_recruitment-test.md · decisions (new Blocking card decision-e38-recruited-before) · How to go on now that the recruited pieces already push toward the target before the fine-tune: (a) amendment with the missing controls (~$0.20), (b) accept and stop the line, (c) redefine the set as pieces nearly off at W0 and rerun. Lead recommends (a).
