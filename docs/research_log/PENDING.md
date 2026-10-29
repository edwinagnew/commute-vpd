# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e47-claim-tests.md (revised after reports/reviews/2026-10-29_review_e47-analysis-check.md) · new E47 card; claims section for (i), (iii) and (v) after the check; for (iv) pending t4c; timeline · five claim tests: (i) stands on two teachers at four starts; (ii) still weakened (rank 8 passes on rows 256-511 only, rank 128 climbs there); (iii) mixed, size alone not isolated; (iv) unsettled as filed (step 1000), the step-10 reading falls by its rule carried by one start, re-read t4c running; (v) falls, plain SGD passes at all six smaller cells (the 0.03 × 1000 fail read on the Adam teacher only)
- 2026-10-29 · reports/plans/2026-10-28_plan_qwen-q0q1.md (Amendment record 2026-10-29) · E44 run line · E40's no-warm-up δW to be remade on a cluster L40 (experiments/qwen_maps/regen.py) and put on W&B, so E44 runs on any H100 without volume ydavm7ct4x; C reruns as the reproduction check; code written, not yet run
- 2026-10-29 · reports/analyses/2026-10-29_analysis_e39-amendment3.md · overview "Where it stands", claims row (teacher training), techniques.html lines 30/53, gaps card "The reference fails the pass check" · add the first-rows caveat to every remaining "every broad fine-tune fails" / SGD-broad-fail statement (left by the v101 flush)
- 2026-10-29 · decisions.md (decision-e46-stage1 → option 3) · Pending decisions (remove card decision-e46-stage1), re-gating "What next" card, overview Line 1 box · user chose one more $0 split-null job before stage 1; stage 1 also waits on E44 and both E45 anchors
- 2026-10-29 · decisions.md (pile rows) · pending decisions; E39 card run line · card decision-pile-rows decided by the user: (a) and (b) on the free cluster; remove the card; E39 run line "E49 row-block job planned"
