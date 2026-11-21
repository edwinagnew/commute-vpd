cell: pending (user, 2026-11-20 ~19:50, "A, reduced redo on built edits and nine pairs"; awaiting `fleet/cell approve 42`)
rival_a: under wording A a built new edit stores its lookup in variables whose M mechanism, even with a gain, loses it, while a built reuse edit's changed variables can take M's mechanism with a gain, predicts all four reuse items (x2, x10, moved, rebuilt) read R_A above both new edits (new_few, new_thin) by more than the split-half band
rival_b: the least-squares split spreads even a pure output gain over many subcomponents and R_A is a share of a circuit mostly left unchanged (R_A >= 1 - f), predicts x10 (f 0.093, its ten-fold output not undone by per-subcomponent gains) reads R_A at or below the higher new edit
rung: premise
smallest_gap: 0.02
spread: 0.0005
n: 2
design: unpaired
unit: known-answer items per side, each one built edit read once on all of D (new side: new_few, new_thin; reuse side: x2, x10, moved, rebuilt)
spread_source: experiments/kstar_vpd/results/summary.json
spread_of: proxy: E226's split-half band of the old R (0.0010, new_thin's two halves; one item's sd about half that), the same items, circuits and windows under the grade this run replaces; this run sets its own band from the two halves before reading the rule
floor: none (the rule is read against the band set in this same job from the six items' two halves; rival a expects min reuse R_A minus max new R_A above the band, rival b at or below 0)
stop: rival b: R_A at this featurization does not read the built labels either, and the lead brings it to the user as a level question (the split, not the wording), without trying another gain fit; rival a: R_A reads the built labels, and the nine pairs' gaps under R_A go into the paper beside the old R
sketch: experiments/kstar_vpd_wording_a/figures/sketch.png
rehearsal: cluster only: fifteen debug-QoS jobs on the MATS cluster ($0, no pod); the scored run is its own smoke: every job reads the do-nothing first (every gain exactly 1, every variable stood in gives KL exactly 0, R_A 1; stops on a miss) and every item checks, before its search, that stood-in unchanged variables leave F's logits exactly and that every change stood in with gain 1 gives M's logits

# Card: does R under wording A read the built labels? (draft)

**What runs.** R under the adopted wording A on E226's six built edits and on E187's nine pairs at matched update
size. For each item, at E226's circuit, we ask how large a share of the circuit's variables can take the original
model's mechanism, each times one gain, all at once inside the fine-tune, and still keep its behaviour within the
circuit's own allowance. The old squared-cosine R is reported beside it. The do-nothing control runs first in every
job.

**Expected.** The reuse edits read 1 or nearly (x2, moved and rebuilt are within the allowance of the original model
already); the new edits read lower, 0.85 to 0.97, because the variables that store the lookup cannot take the original
mechanism. The risk is x10: the split spreads a gain over many parts, so it may read as low as 0.907.

**Cost.** About 18 L40-hours on the cluster in fifteen jobs of under 2 h, $0, about 6 h of wall time.

**Decision it serves.** Whether the paper reports R under A as reading the built labels at the VPD level, and the
nine pairs' gaps under it.

Plan `reports/plans/2026-11-20_plan_r-under-a.md`.
