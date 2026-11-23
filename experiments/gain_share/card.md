cell: P1.58
rival_a: E62's copying map is mostly the base model's own variables turned up or down, its lookup is not, and the best gain-only fit can find that at either level of variables; this predicts the gain-only share of the copying map is above the lookup's by at least 0.1 (mean over seeds 0-2) both with one gain per neuron and head and with one gain per VPD subcomponent and remainder
rival_b: VPD spreads the one-head rescale that the copying map makes over many subcomponents shared between heads, so no setting of their gains rebuilds it, while one gain per head can; this predicts that with one gain per VPD subcomponent both maps' shares are under 0.3 and within 0.1 of each other, while with one gain per neuron and head the copying map is above the lookup by at least 0.1
rung: strategy
smallest_gap: 0.1
spread: 0.0487
n: 3
design: unpaired
unit: E62 fine-tune seeds per map (copy_s0-2, lookup_s0-2), one gain-only fit each per level
spread_source: experiments/induction_cue/results/run.json
spread_of: proxy: the share of F's gain lost with head (2,4) switched off, sd over the copying map's seeds 0-2 (states.copy_s{0,1,2}.share: 0.563, 0.545, 0.471), which has this reading's units (a fraction of the same fine-tune's trained-answer gain on E62's read rows) and the same seeds; the lookup's sd on the same reading is smaller
floor: MIN_GAIN, 1 log-odds of F's trained-answer gain on the held-out rows before a share is read (E236's); E62's fine-tunes gain about 12.6 (copy) and 15.2 (lookup) log-odds (experiments/gain_only/results/items/e62_*_s0.json refs), so both rivals expect all six read; a state under it is read by its KL-recovered share only
stop: if the copying map and the lookup read within 0.1 of each other at the neuron-and-head level too, the gain-only share does not tell the user's worked example apart where head ablation does, and option 2 is not carried past this prototype
sketch: experiments/gain_share/figures/sketch.png
rehearsal: cluster only: one job on one L40 in the cluster's debug QoS (under 30 min, no pod); the scored run is its own smoke, stage 0 (gains of 1 change no bit at either level, the VPD remainder identity, the exported change equals the hooked one, a do-nothing reads KL 0) and stage 1's controls run first and each miss stops the job
