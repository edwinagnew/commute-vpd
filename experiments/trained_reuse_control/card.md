cell: P1.4
rival_a: The definition reads what an update does inside the model, so a fine-tune that copies a reuse edit's outputs closely does so by strengthening the same parts, which predicts that at the neuron featurization (E_b over K, remainders in, as E277 filed it) the imitators of x2 read at least 0.2 above the imitators of new_few of the same seed, on every seed where both pass the closeness bar
rival_b: The instrument reads low on any trained update whatever the update does (P1.59's finding, the trained fine-tunes at 0.014-0.038 in E277), which predicts that the imitators of x2 read within 0.2 of the imitators of new_few of the same seed, both near the trained fine-tunes and their matched randoms, even where both pass the closeness bar
rung: premise
smallest_gap: 0.2
spread: 0.011
n: 3
design: paired
unit: seeds (0-2) of the imitator recipe, the x2 imitator paired with the new_few imitator of the same seed
spread_source: experiments/needed_neurons/results/summary.json
spread_of: this reading (E_b over K at the neuron featurization on E277's trained fine-tunes, sd of the two seeds per task, largest 0.011 on repeat_real)
floor: an imitator is read only when its leftover TV(fine-tune, target) / TV(fine-tune, M) on D at step 3000 is at most 0.1924, E139's measured leftover at the end of the same recipe (results/amend2_always_on_s*.json, seeds 0.1922, 0.1918, 0.1924), with both gaps at least E139's 0.02 floor; and |K| >= 100 (E277). Both rivals expect the new_few imitators to pass (eight token associations, a target gap near 1); x2's are the risk (301 training windows, a mild target), and fewer than 2 seeds with both passing is no call
stop: If the x2 imitators pass the closeness bar on at least 2 seeds and read within 0.2 of the new_few imitators (rival B), the paper states that a fine-tune that copies a reuse edit's outputs to E139's closeness does not read as reuse at neurons, and no further imitation controls are proposed; if x2 or new_few, read again, misses its prior E_b over K, every E_b reading stops until the cause is found
sketch: experiments/trained_reuse_control/figures/sketch.png
rehearsal: E___ rehearse: on cluster debug L40s ($0): the train jobs run first as their own rehearsal (the first 30 steps project the time and stop over 6,000 s; R1's early checks at steps 1, 2, 3, 10 and 300), and the check job reads x2 and new_few at neurons at full read sizes, measuring time per item; before the H100 pod, one 40-min debug job runs `--stage vpd`: its stage 0 and both VPD repeats (x2, new_few) at full read sizes (about 34 min at E268's times), measuring time per item and peak memory, and writes the repeat file the pod then skips; the four seed-0 VPD items go to one H100 pod (about 66 min of reads plus set-up, about $4-6)

## What runs (cell P1.4 run 2 of 3; reopened 2026-11-23 by whip-lead under the user's standing authority)

Plan: `reports/plans/2026-11-24_plan_p1-4-run2-trained-reuse-control.md`. Filed rules: `results/predictions.json`.

Two built edits whose label is known by construction (E214): **x2** doubles the writes of eight copy neurons in layer 1
(pure strengthening: reuse) and **new_few** writes eight new token associations (new). For each, three ordinary
fine-tunes of the 24 decomposed matrices are trained to copy the edit's next-token outputs on its own task windows, with
E139 amendment 2's recipe (3,000 steps, peak 1e-5, warm-up 300, cosine). Each imitator is read like any update, on its
target's D, beside a random update matched to it per tensor (E277's construction). The scored reading is E_b over K at
neurons; VPD (seed 0, on one H100) is descriptive.

The closeness bar comes first: an imitator that lands far from its target is no reading, because then its outputs are
not the edit's. The bar is the closeness E139's recipe reached on this model, not a chosen number.
