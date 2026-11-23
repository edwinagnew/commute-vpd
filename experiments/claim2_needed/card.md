cell: P2.62
rival_a: The needed set K and the mean-ablation circuit K* rank these states' write shares alike, which predicts every reading's rank correlation and partial rank correlation given KL with E_b moving by less than 0.2 from K* to K, every built pair's E_b gap keeping its sign, and E_b over K splitting the ten copy seeds from the ten prior-matched cipher seeds as the cheap readings do
rival_b: K and K* disagree on the built-pair and prior-matched states as they did on repeat-task RL, which predicts at least one reading's rank correlation or KL partial with E_b moving by 0.2 or more, or at least one built pair's E_b gap changing sign, from K* to K
rung: premise
smallest_gap: 0.2
spread: 0.04
n: 39
design: paired
unit: states (each saved state read over K* and over K; n is the procedural table's trained states, 9 pairs and 20 prior-matched seeds in the other tables)
spread_source: experiments/kstar_needed/results/summary.json
spread_of: proxy: E_b, seed sd of E220's size-matched RL arm over K* (0.161, 0.139, 0.217; E268's card spread); a rank correlation moves only when states' E_b ranks move, and states within this spread of each other can swap ranks
floor: none (descriptive)
stop: every reading's correlation and KL partial with E_b moves by less than 0.2 and no pair gap changes sign: the paper keeps claim 2's numbers, quoted over K, and no further K work on claim 2 is proposed
sketch: experiments/claim2_needed/figures/sketch.png
rehearsal: cluster only: K scans on saved states run on MATS cluster debug QoS, $0, stage 0 identity-swap checks first as in E268
descriptive: whip-lead on the user's approval 2026-11-22 ~18:15; cell done_when: 'Report which numbers in the results picture change'

What this re-reads (cell P2.62, done_when verbatim): "Once P1.54 lands, recompute claim 2's comparisons from saved files
($0): re-gating share, steering, effective rank, neuron, crosscoder, ADL and KL against E_b over K, on built pairs,
prior-matched pairs and procedural states, with the KL partials. Report which numbers in the results picture change."

E268 (P1.54) already holds E_b over K for every procedural and repeat-task state, so that table is CPU only. The nine
built pairs' 76 states, the other 18 of E248's 94 items, and seeds 3-9 of E254's prior-matched copy and cipher have no
K yet: `run.py` scans them with E268's settings and code path (`commute.needed.scan`, 4 derangements x 64 windows,
last read position), stage 0 first.
