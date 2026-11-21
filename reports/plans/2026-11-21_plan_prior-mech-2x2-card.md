cell: P2.54
rival_a: the copy-head label follows the mechanism (copy answers come from the window through head (2,4)) predicts copy and E62's copy lose ~0.5 of their gain with (2,4) off while the prior-matched cipher, cipher, rank1 and elicit lose ~0
rival_b: the head label follows the base prior (high-prior answers ride the same head whatever was learned) predicts the prior-matched cipher loses about as much as copy (~0.4-0.5), and the low-prior E62 copy loses little
rung: premise
smallest_gap: 0.3
spread: 0.06
n: 3
design: unpaired
unit: seeds
spread_source: experiments/induction_cue/results/amend5.json
spread_of: proxy: E62 copy share of gain lost with (2,4) off across 3 seeds (0.44-0.56), the same reading on the same 67M recipe
floor: MIN share 0.2 (share_new_max in predictions.json); under A copy reads ~0.5 and both ciphers ~0; under B prior_cipher reads ~0.4-0.5
stop: copy's share under 0.2 and E62 copy's under 0.2: the head does not label reuse on these tasks, so the 2x2 needs another intervention
sketch: experiments/prior_mech_2x2/figures/card_sketch.png
rehearsal: cluster only: run 1 is a ~6 min read of saved states on the MATS debug L40, $0, no pod; the control (M's induction copying, zero update) runs first and alone and exits on a miss
