cell: P1.56
rival_a: The write share measures whether a fine-tune strengthens what M already computes, whatever the basis, which predicts that at the top-activating-neuron featurization the strengthening built edits (x2, x10) read E_b over K at least 0.2 above the new edits and the randoms, as at VPD (0.76 vs 0.01-0.02) and at the SAE (0.95-0.97 vs at most 0.02)
rival_b: A gain on one VPD subcomponent is spread over many neurons whose single-neuron gains do not each match M's write, which predicts that at the neuron featurization the strengthening edits read E_b over K within 0.2 of the new edits and the randoms (the order collapses at this basis)
rung: strategy
smallest_gap: 0.2
spread: 0.04
n: 3
design: paired
unit: seeds per trained arm (the same reading at a second featurization); built edits are one state each
spread_source: experiments/rb_states/results/summary.json
spread_of: proxy: E_b over K* at VPD, seed sd of the E220 rl arm (0.16, 0.14, 0.22), the same write share read at another featurization
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23, filed with E273); both rivals expect |K| of hundreds or more at every built edit
stop: If the strengthening edits read within 0.2 of the randoms at the neuron featurization, §4 lists the neuron basis as one where the write share does not separate strengthening from new, and no further neuron-basis E_b runs are proposed
sketch: experiments/needed_neurons/figures/sketch.png
rehearsal: E277 rehearse: on a cluster debug L40 ($0): stage 0 (identity swap gives zero flips; do-nothing E_b over K is 1) and one built edit at full read sizes (256 pairs, all read positions), measuring time per item and peak memory; the scored items then go to cluster debug if it starts at once, else to H100 pods (user 2026-11-23: "if it would be quicker to use runpod, do so")
descriptive: user, 2026-11-23, task file via writeup-lead: the reading is wanted "for §4's list of featurizations"; result to redteam-lead for a verdict

## What runs (E277 below, cell P1.56; user request 2026-11-23)

The needed set K (single-variable interchange on D, E268's rule: a variable is needed when swapping its value for the
one it takes on another input of D changes F's top token at a read position) and the write share E_b over K, at E229's
top-activating-neuron featurization: at each MLP site a neuron is a feature on a token when its output is above its base
p97 on Pile text (E217's saved percentiles), everything else at the site is one exact remainder; each attention head is
a variable (E217's `neuron_head` level with a remainder). The rule is fixed from M.

Order: stage 0 alone (identity swap: 0 flips; do-nothing: E_b over K exactly 1); then the seven E247 built edits (x2,
x10 strengthening; rebuilt; new_few, new_thin; random_size, random_kl as the control); then E229's 20 trained
fine-tunes (5 pairs x 2 seeds) each beside a random update matched to its per-matrix size. |K|, median top-1 minus
top-2 margin at the read positions and E_b over K are reported for every item; |K| < 100 reads "too small to read".
