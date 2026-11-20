cell: P2.47
rival_a: the share separates the nine built pairs because Gamma, trained to find what each position needs, switches on base subcomponents the fine-tune newly relies on; a gate that keeps whichever subcomponents fire hardest does not see that predicts the top-k gate (k matched to Gamma's mean on-count) keeps at most 6 of the 9 pairs apart in the built direction (P1 'falls to the neuron level'), as neurons for half the gain do
rival_b: the separation lives in the subcomponents themselves: a fine-tune that reuses the base moves its inputs along the read directions V_c of existing subcomponents, so any gate that follows their activations re-gates the same ones predicts the top-k gate keeps 7 to 9 of the 9 pairs apart in the built direction (P1 'holds above')
rung: strategy
smallest_gap: 0.1
spread: 0.0314
n: 2
design: unpaired
unit: seeds (saved fine-tunes) per side of a pair, 2 to 8 a side (68 fine-tunes over the nine pairs); the count read is pairs apart out of nine
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the log-odds share over five 64-row batches of one fine-tune (.stage1_step10.sd_r = 0.0314), the same share in the same units; the pair rule is ordinal (every seed of one side beyond every seed of the other), so the count of pairs apart is read against the neuron version's 6; smallest_gap is a reuse-minus-new gap in share units (the scorecard's share gaps run 0.39 to 0.69, so a gate whose gaps shrink under 0.1 has lost the separation)
floor: MIN_GAIN 1 log-odds of target gain (local_faith.MIN_GAIN), applied to every share; every saved fine-tune of the nine pairs is above it (the saved shares are all finite), so neither rival expects a pair 'not read'
stop: P1 'holds above' with the gamma gate reproducing every saved share: the separation is carried by the subcomponents, not the trained gate function, and section 5.2 reports the share's column as a unit result beside the neuron columns; no further gate variant is run for the paper
sketch: experiments/topk_gate/figures/sketch.png
rehearsal: cluster only: one debug-QoS L40 job (~12-15 min, the forwards E136 ran in ~5 min times ~2.5), with the do-nothing control first and the gamma-gate reproduction of every saved share as its in-job control; no pod

## What runs

The 68 saved fine-tunes of E187's nine built pairs, on E136's read rows, each read once with five gates in the same
forwards: Gamma's own masks (the positive control, must reproduce the saved share), Gamma's masks binarised at 0.5,
and the top-k gate at k matched to Gamma's mean on-count (scored), at half and at double. The top-k gate keeps, at
each position, the k alive subcomponents of a matrix whose inner activation |x . V_c| is largest in the clean forward
of the model as it stands. Do-nothing control first: a zero change reads every numerator exactly 0. Cluster debug
QoS, $0.
