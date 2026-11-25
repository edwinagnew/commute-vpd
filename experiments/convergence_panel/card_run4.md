cell: P2.66
rival_a: arxiv, german and spanish s0 sit under the floor only because 300 steps was too short, which predicts that retraining to 1200 steps under the same recipe gives every seed a save with read-row gain >= 1, so all 12 tasks enter every reading
rival_b: the recipe cannot teach arxiv and german past the floor (gain flat from step ~110 to 300), which predicts that some seeds of arxiv or german s2 reach no save with read-row gain >= 1 by step 1200, so the panel stays at n = 11 or under unless the user changes the recipe
rung: parameter
smallest_gap: 0.1
spread: 0.11
n: 7
design: unpaired
unit: task-seeds retrained (arxiv s0-2, german s0-2, spanish s0); the reading is each seed's earliest saved step with read-row gain >= 1, and its gain
spread_source: experiments/convergence_panel/results/wide/read/german.json
spread_of: proxy: german's read-row gain across seeds at step 300 (0.56-0.78), the same reading in the same units (descriptive run: no bar check; the spread only fills the form)
floor: MIN_GAIN 1 log-odds on the read-row gain (results/wide/predictions_run4.json); rival a expects every seed to cross by 1200, rival b expects arxiv and german s2 not to
stop: the stage-0 replay control misses (retrained step-300 gain off the saved step-300 file by more than the measured floor): the probe stops, nothing is picked, and the lead reports to whip-lead
sketch: experiments/convergence_panel/figures/hope1.png
rehearsal: cluster only: four debug QoS jobs on cluster L40s at $0, each under 45 min; the later H100 SXM E_b reads reuse E272's eb.py and its stage 0 (E272's rehearsal on record, job 63920)
descriptive: user 2026-11-24 via writeup-lead and whip-lead: "ask whip-lead for the run, get everything to 12"; whip-lead approved run 4 at ~17:1x and reopened P2.66; reported as a description
