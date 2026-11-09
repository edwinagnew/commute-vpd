# Does the share still separate built pairs without a trained decomposition?

*compare-lead, 2026-11-09, cell P2.8, due 11-12. One run, free queue, saved updates only. Sides are labelled as each pair was built; no verdict on reuse. Provisional.*

I replaced the trained decomposition with two that need no training, each matrix's singular value decomposition and the same with its directions randomly rotated, and read the re-gating share on the saved fine-tunes with a fixed gating rule. On all five pairs that could be scored, the share's separation between the two sides disappeared, so that separation depends on something the untrained versions lack. This is evidence that the trained decomposition matters to the share, limited to this one gating rule, and the untrained versions reproduce the model much less closely than the trained one does.

**The figure** is `experiments/share_untrained/figures/gaps.png`: the gap between the side built to reuse and the side built to be new, per pair and decomposition, with the copies behind each.

On the five scored pairs the trained gap is 0.51 to 0.69 of the gain. On the singular value decomposition it runs from 0.05 below zero to 0.07 above, so each pair loses between 0.44 and 0.68, well past the 0.1 that would change our minds. The filed reading is that the separation comes from the trained decomposition, on five of five pairs. The randomly rotated version also reads little, from 0.12 below zero to 0.16 above. Over all saved updates the share itself averages 0.50 on the trained decomposition and 0.07 on the singular value one, so the untrained reading is low everywhere and not only flat between the sides.

**The limit on that reading.** The untrained versions replace both the pieces and the gating, and the gating is a rule chosen before reading: at each position keep the few components with the largest contribution, as many as the trained decomposition has switched on, between 1 and 30 per matrix. With those gates the original model is reproduced worse: its masked answer log-odds sit 3.1 below the original on average on the singular value decomposition, and up to 10.6 below on one update, against 24.1 below for the rotated one and 0.8 above for the trained one. So a low share here could mean the pieces lack what the share needs, or that this rule picks poorly. The run cannot separate those, and it tests one rule.

**The twin pairs,** fine-tunes on real labels against the same text with labels shuffled, are described only, at two copies per side. They show no gap on any reading: at most 0.04 from zero on the trained and on the singular value decomposition. Three more pairs were not read because their fine-tunes were never saved.

**Controls.** The decompositions add back to the matrices exactly, all gates on reproduces the model, a zero update reads exactly nothing, and the trained share reproduced its saved value on all ten pairs. The first job stopped on that reproduction: the code had computed the original model's reference at the fine-tuned weights. Once fixed, the reproduction passed, so that was the cause.

**What it is blind to.** The share here says only how far the masked model moves with the state's gates, not whether a component is interpretable. The trained gates are continuous and the rule's are 0 or 1. The two open-pair rows share one side, so the five pairs are not fully independent.

## Departures from the literature

Replacing a trained decomposition with a singular value decomposition, and gating by largest contribution, are this cell's own stand-ins, not a method from the parameter decomposition papers.

## Not run

Any other no-training gating rule, including continuous gates or a larger number kept per matrix. The three pairs whose fine-tunes were not saved, and the third copy of the class-rule pair. The share on an untrained decomposition with more faithful masking.
