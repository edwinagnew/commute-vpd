cell: P1.40
rival_a: under wording A a built new edit stores its lookup where M's mechanism with a gain cannot write it, while a built reuse edit's changed latents can take M's mechanism with a gain, predicts all four reuse items (x2, x10, moved, rebuilt) read R_A at SAE latents above both new edits (new_few, new_thin) by more than the split-half band
rival_b: a base-only SAE puts what its dictionary lacks into one remainder per site, so a new edit's whole new write is one variable of K* and R_A counts variables, predicts the new edits read R_A at or above the lowest reuse item, with their remainder share above 0.5
rung: strategy
smallest_gap: 0.02
spread: 0.0005
n: 2
design: unpaired
unit: known-answer items per side, each one built edit read once on all of D (new side: new_few, new_thin; reuse side: x2, x10, moved, rebuilt)
spread_source: experiments/kstar_vpd/results/summary.json
spread_of: proxy: E226's split-half band of the old R at VPD (0.0010 on new_thin's halves; one item's sd about half that), the same items and windows at another featurization; this run sets its own band from the two halves before reading the rule
floor: none (the rule is read against the band set in this same job from the six items' two halves; rival a expects min reuse R_A minus max new R_A above the band, rival b at or below 0)
stop: rival b, or any order other than every reuse item above both new edits: the key tasks and pairs are not read, and the SAE reading goes into the paper as a featurization that cannot see a new write held in its remainder
sketch: experiments/sae_r_a/figures/sketch.png
rehearsal: E245 rehearse: cluster debug L40, do-nothing and x10 at the full read sizes with the search's evaluation cap at 50; measures time per evaluation, peak GPU memory, the positive-control floor (every variable stood in with gain 1 against M's logits) and that unchanged sites' gains are exactly 1

# Card: does R under wording A read the built labels at SAE latents?

**What runs.** The same question E242 asks at VPD, at E234's base-only SAEs on the attention and MLP outputs: what share of the behaviour's circuit can take the original model's mechanism, each times one positive gain, all at once, and keep the behaviour within a tenth of how far the fine-tune moved it. Do-nothing first, then the six built edits, then E237's 15 key items, then E187's nine pairs only if time allows.

**Expected.** The reuse edits read near 1. The open question is the new edits: the base SAE cannot write their new neurons, so their change lands in one remainder per site, and a share of variables may barely notice it (rival b). The remainder share is read beside R_A to show this.

**Cost.** Cluster only, $0, about 2 h of L40 time for the built edits and 3 h for the key items.

**Decision it serves.** Whether the paper's R_X row for SAEs reads the built labels, or is reported as blind to a new write held in its remainder.

Plan `reports/plans/2026-11-21_plan_r-under-a-sae.md`.
