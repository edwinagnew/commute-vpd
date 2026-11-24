cell: P1.58
rival_a: a gain-only fit to F recovers F only where F runs on M's own variables at a new strength, which carries over to inputs the fit never saw, while run 1's ~0.96 on the built new edits was imitation of the fitted cues' answers; this predicts that on unseen cues and keys x2 and x10 at the neuron level keep a KL share of at least 0.9, both built new edits fall to a share of at most 0.3, and then E62's copying map reads at least 0.1 above its lookup (mean over seeds 0-2)
rival_b: thousands of gains can imitate any narrow change, and the imitation carries over to unseen cues of the same task; this predicts that the built new edits keep a share above 0.3 on unseen cues, so the controls do not separate and the run stops before E62
rung: parameter
smallest_gap: 0.1
spread: 0.0217
n: 3
design: unpaired
unit: E62 fine-tune seeds per map (copy_s0-2, lookup_s0-2), one gain-only fit each per level
spread_source: experiments/gain_share/results/score.json
spread_of: proxy: run 1's share of E62's copying map at the VPD level, sd over seeds 0-2 (0.9866, 0.9433, 0.9664; 0.0054 at the neuron level), the same reading (a fraction of the same fine-tune's trained-answer gain on E62's read rows) on fitted rather than unseen keys, at about half the read rows here, which may raise it by up to sqrt 2
floor: MIN_GAIN, 1 log-odds of F's trained-answer gain on the read rows before a share is read (E231's); F's gain was 12.5-15.4 (E62) and 16.5-16.8 (new edits) on all keys in run 1 and F is trained on every key, so both rivals expect every unseen side read; x2 and x10 gain under 1 and are read by their KL share in both rivals
stop: if the controls do not separate on unseen cues at the neuron level (x2 or x10 under 0.9, or a new edit's share above 0.3), the gain-only share is not carried past this run in any form tried here, and E62 is not read
sketch: experiments/gain_share/figures/sketch_unseen.png
rehearsal: cluster only: two jobs on cluster L40s in the debug QoS, one per level (neuron about 30 min, VPD about 60 min, each under the 2 h cap), no pod; the scored run is its own smoke: stage 0 and the exact controls x2 and x10 run first and each miss stops the job
descriptive: whip-lead 2026-11-23, P1.58 reopened for the unseen-cue test: one run, reported as a description; the gate is a stop rule, the gap is read but carries no claim at the bar, and no reuse or new verdict is given while the definition is under revision
