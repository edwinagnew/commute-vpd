# Does steering separate copy from lookup as widely as the share?

*compare-lead, 2026-11-09, cell P2.5, due 11-12. One run, free queue, saved updates only. Labels are "copy" and "lookup" as each fine-tune was built; no verdict on reuse. Provisional.*

I gave steering its fair form on the copy and lookup fine-tunes, with one vector per key and a strength tuned on separate windows, and compared its gap between the two with the re-gating share's. The share's gap stayed wider than steering's widest, so on this pair the share keeps an edge under the rule filed before the run. That edge is over one kind of steering only, because one vector per key reads copy and lookup alike, so what steering says about this pair depends on the vector, and neither reading settles whether either side is reused.

**The figure** is `experiments/steering_copy_lookup/figures/steering_copy_lookup.png`: share of the gain, three seeds per side, 100 scored windows.

The share reads copy above lookup by 0.69 of the gain. One pooled steering vector does the same in kind, with a gap of 0.41: copy reads 0.52 to 0.59 and lookup 0.14 to 0.16. Giving each key its own vector raises lookup to 0.63 to 0.69 and leaves copy at 0.61 to 0.65, so that gap all but vanishes, at 0.02 in lookup's favour. The filed rule takes the wider of steering's two gaps, the pooled one, and the share beats it by 0.27, against a smallest gap that matters of 0.1. The largest seed sd of any side and method is 0.04. A likely reason for the two kinds' difference, untested here, is that one vector cannot hold 32 different answers while a vector per key can.

In log P, the old columns' unit, the same pattern holds: the pooled gap is 0.50, and the per-key gap is 0.08 in lookup's favour.

**Controls.** A zero update and strength zero both read exactly nothing, and all six shares reproduced their saved values exactly. Random vectors of the same norm read below zero, as they should not help; the per-key one on lookup is 0.12 below zero, outside the band of 0.1 filed for it, so the per-key lookup reading carries that mark. Steered, the rest of the window moves little: off-target KL is 0.003 to 0.016, against 0.06 to 0.09 for the fine-tunes themselves. Vectors are specific to their side: lookup's per-key vectors push copy's answers down, by 0.31 to 0.38 of the gain. Pooled vectors at the second block agree across seeds, 0.89 within copy and 0.85 within lookup, and 0.31 between the two.

**Where it missed what I expected.** I expected pooled copy at 0.7 to 0.9 in log-odds, by analogy with an earlier pair, and it read 0.52 to 0.59. In log P it reads 0.73 to 0.80, so the analogy was probably in that unit. I also expected the per-key vector to read like the pooled one on copy, whose answer varies within a key; it reads somewhat higher.

**What it is blind to.** The steering share says only how far an added vector moves the answer, not whether it does so by the fine-tune's route. The strength and layer were chosen on windows the per-key vectors were fitted on, which favours per-key. Steering's vector was picked per state, while the share has no such choice.

## Departures from the literature

None: the share and the steering share are this project's measures. Steering with a mean difference at one layer follows activation addition; the per-key form and the strength grid are additions asked for here.

## Not run

The neuron version on this pair, which this cell does not read. Cross-fitting of the strength choice. Any label under the definition being revised. The states' spread on other marker tokens.
