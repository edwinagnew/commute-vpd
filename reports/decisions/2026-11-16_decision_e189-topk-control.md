# P2.29, due 11-18: E189's top-K control failed; rerun with a K grid, or close as it stands?

**Question.** The Goodfire-style run on the 67M built pairs stopped at its first stage. One of its two apparatus checks
missed, so no fine-tune was read. The cell's one run is spent. Should a second run be allowed, with the sparsity reading
changed, or should the cell close as it stands?

**What was read** (job 60130, debug L40, 54 s, $0; `experiments/neuron_jobs/results/stage0.json`). Everything below was
read on one fine-tune, the copy task's first seed.
- **Job retention check: passed.** It asks how many of a neuron's 20 strongest Pile windows stay the same after a
  change. Making the moved neurons fire harder kept 0.67 of their windows. Swapping their jobs kept 0.10. On the copy
  fine-tune itself the neurons it moved most kept 0.75.
- **Sparsity check: failed.** The model ran with each block keeping its 240 strongest neurons per token (7.8%,
  Goodfire's share). A known update was placed only on the neurons that are most often among the 240. That update
  should survive the mask almost whole. Only 0.17 of its gain survived, against 0.01 under 240 random neurons. The
  filed check needed a gap of 0.4. The copy fine-tune itself kept 0.14 of its gain, against 0.02 at random.
- **Likely reason (not tested).** On this small model a 7.8% mask, applied block after block, probably changes the
  model so much that any gain mostly vanishes. Goodfire's model is much wider (4,096 per layer, 320 kept). The control's
  0.4 gap was a guess, not a measured number.

**Options**
1. **Close the cell as it stands (no run).** The report says retention works as an instrument but was read on one
   fine-tune only, and that top-240 masking keeps little of any gain on the 67M model, so the second baseline is
   unreadable at Goodfire's share. Cost $0. The gate page says the Goodfire baselines were not read.
2. **One more run (recommended): the same job with the sparsity reading at three sizes, 240, 768 and 1,536 of 3,072.**
   The sparsity check is recorded at each size rather than stopping the job. The retention reading runs as filed, since
   its check passed. This tells us whether some K keeps the known update's gain, and both readings on all nine pairs.
   Cost $0 on debug, about 10 min. It needs a new 1-run cell (an amendment is a run).
3. **One more run, retention only.** Drop the sparsity reading. This gives the per-neuron job column on all nine pairs.
   Cost $0, about 5 min.

**Recommendation.** Option 2: the retention reading is ready, and one grid tells us whether the sparsity reading can
work on this model at all.

Evidence: `experiments/neuron_jobs/results/stage0.json`; plan `reports/plans/2026-11-16_plan_p2-29-neuron-jobs.md`.
