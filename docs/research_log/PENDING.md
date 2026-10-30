# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead: E55 card: C1b result · reports/analyses/2026-10-31_analysis_e55-c1b.md, notebooks/E55_pcfg_anchors.ipynb (section "Stage C1b") · E55 card (figures experiments/pcfg_anchors/figures/c1b_{readings,scores}.png), claims table, timeline · with cut-offs set on held-out seeds, pruning revival, reverse fine-tuning and steering each label all 21 scored states right (needs 17; best recipe labeller 15) and 12/12 per run: all three pass. But every state performs one of two tasks perfectly, so the answer-only probe and update size read within a recipe also get 21/21: the set shows the cheap methods tell count a + 1 from count b, not that they read reuse; VPD (C2) can at best tie on it. C1's misses were its guessed cut-offs.
- 2026-10-31 reuse-lead: E55 card "Running / Due to run" line + timeline: independent check (reports/reviews/2026-10-31_review_e55-c1b-check.md) found the C1b report sound, fixes applied (cf8d46e). Next on the reuse line (Fable's advice, reports/plans/2026-10-31_advice_e55-after-c1b-FABLE.md, adopted): run pruning, reverse fine-tuning and the probe on E58's 12 open-pair 67M states ($0, cluster), to see whether VPD's granularity limit is shared by every instrument; C2 (VPD on the pcfg set) deferred, since on a set where update size already reads 21/21 VPD can at best tie.
