cell: P1.23
rival_a: The side built to reuse sits closer to something already on the original model's list, so a few of those names close more of its gap than of the matched new side's, which predicts that on fresh seeds 2 and 3 of all five trained pairs the reuse side's closure at 16 names (list all) exceeds the new side's by more than 0.029, on all ten pair-seeds, under the full-distribution reading
rival_b: The greedy search over the list closes any fine-tune that moves a few output logits about as well whatever it reused (E184's case and repeat pairs sat within 0.016 of each other), which predicts at least one of the ten fresh pair-seeds, most likely case or repeat, with a reuse-minus-new closure gap at 16 names of 0.029 or less
rung: parameter
smallest_gap: 0.029
spread: 0.0103
n: 10
design: unpaired
unit: pair-seeds per side (one fresh fine-tune per side of a trained pair, seeds 2 and 3, five pairs)
spread_source: experiments/e184_curves/results/curves.json
stop: The filed rule fails under the full-distribution reading on fresh seeds (some pair-seed at or under the bar): the curve reading is reported as not telling every trained pair apart, and no further search on it is proposed from this cell
sketch: experiments/curve_test/figures/sketch.png

The user's request (2026-11-16), verbatim: "New cell, 2 runs, due 11-18: a filed test of the E184 curve reading on fresh
fine-tunes. Fourth wording unchanged, still tested, not adopted. Train seeds 2 and 3 of the five matched pairs (copy/cipher,
identity/random map, case, repeat, parity twin/memorised fact) with their original recipes, then run E184's list search on
them: list all, up to 16 names, no stop at 0.95, no free-direction fits. File the rule before any read: at 16 names, the reuse
side's closure exceeds the new side's by more than the largest same-side seed gap in E184, on every pair and seed. Read it
twice and report both: closure on the full next-token distribution, and closure on the answer tokens only (chosen after
seeing E184, say so). Controls first and alone: one shifted neuron, one triggered neuron, random bit per window, as in E184.
RunPod is fine. Write-up via /writeup, with the E184 curves beside the new ones."

Bar: the largest same-side seed gap in E184 at the last name, list all, is cipher24 (seed 0 0.4260, seed 1 0.4555, gap
0.0295; `experiments/e184_curves/results/curves.json`, `.items[].lists.all.best[16]`). Spread: the root mean square over
the ten sides of |seed 0 - seed 1| / sqrt(2), the sd of one seed's closure, from the same file. The rival lines name the
full reading; the answer-position reading uses the same bar (measured on the full reading only) and is reported beside.
Reuse and new sides as E184 filed them (`experiments/simplest_intervention/run.py`, OLD_PAIRS and R1): copy24, ident32,
case_real, repeat_real and the parity twin reuse; cipher24, rand32, case_twin, repeat_twin and the memorised fact are new.
