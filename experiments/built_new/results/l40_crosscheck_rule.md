# E304: L40 cross-check rule (filed 2026-11-24, before the cross-check runs and before any E304 reading is opened)

Why: RunPod had no L40 (two refusals, 400 "no instances"), so every E304 pod is an L40S. VPD, neuron and SAE readings
are card-dependent through SVD signs (friction 1085), and all of E301 and Fig 2's P1.62 reads are on L40. whip-lead
(2026-11-24): "L40S with one L40 cross-check item and a filed bar from its measured gap".

**Amended 2026-11-24 (whip-lead, before the cross-check ran or any E304 reading was opened): the bar and the items
below replace the first filing (0.03, new_few only), kept at the end for the record.**

What runs: the first passing draw's `new_few` AND its `rand_new_few` read at VPD, top neurons and SAE latents on a cluster L40 (`cluster/built_new.sbatch`,
`--out l40check`), the same code, settings and PRED_SHA as the pods.

Reading: per featurization and item, gap = |E_b_K (L40S pod) - E_b_K (L40 cluster)|, with E_b_K as P1.62 reports it
(leftovers counted); also reported: |K| on each card and the overlap of the two K sets.

Rule (6 gaps: 2 items x 3 featurizations):
- every gap <= 0.0019 -> the L40S reads are used as they stand, and the page states the six gaps;
- any gap > 0.0019 -> no E304 dot goes into Fig 2 from L40S; the 12 states are read again on cluster L40 (overnight) and
  the page reports both cards.
Source of 0.0019: half the smallest new-vs-own-random distance among Fig 2's current built rows
(experiments/fig2_standard/results/definition_rows.csv, field E_b_K): VPD new_few 0.012698 vs its random 0.008936,
distance 0.00376 (the others: VPD new_thin 0.0155, top new_few 0.0283, top new_thin 0.0140, SAE new_few 0.0249, SAE
new_thin 0.0051). whip-lead's stated reason for the bar is that a card gap must not be able to swap a new edit with its
random; "<= 0.01 (about half the smallest new-vs-random distance)" quoted 0.01, but half the measured smallest distance
is 0.0019, so the stricter number, which matches the stated reason, is filed. No measured L40-vs-L40S floor for E_b_K
exists in the repo (friction 1085's known-value control was L40 vs H100).
The bar is not changed after the cross-check is read.

First filing (superseded before any read): new_few only; every gap <= 0.03 (the card's smallest_gap).
