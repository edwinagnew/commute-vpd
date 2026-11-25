# E304: L40 cross-check rule (filed 2026-11-24, before the cross-check runs and before any E304 reading is opened)

Why: RunPod had no L40 (two refusals, 400 "no instances"), so every E304 pod is an L40S. VPD, neuron and SAE readings
are card-dependent through SVD signs (friction 1085), and all of E301 and Fig 2's P1.62 reads are on L40. whip-lead
(2026-11-24): "L40S with one L40 cross-check item and a filed bar from its measured gap".

What runs: the first passing draw's `new_few` read at VPD, top neurons and SAE latents on a cluster L40 (`cluster/built_new.sbatch`,
`--out l40check`), the same code, settings and PRED_SHA as the pods.

Reading: per featurization, gap = |E_b_K (L40S pod) - E_b_K (L40 cluster)| on that item, with E_b_K as P1.62 reports it
(leftovers counted); also reported: |K| on each card and the overlap of the two K sets.

Rule:
- every gap <= 0.03 (the card's smallest_gap, the smallest difference that would change a reading of Fig 2's built block)
  -> the L40S reads are used as they stand, and the page states the three gaps;
- any gap > 0.03 -> no E304 dot goes into Fig 2 from L40S; the 12 states are read again on cluster L40 (overnight) and
  the page reports both cards.
The bar is not changed after the cross-check is read.
