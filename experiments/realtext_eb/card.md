cell: P2.64
rival_a: The real-text fine-tunes built to need what M holds (topic label, lowercase, French) keep more of M's mechanisms on the variables they need than their twins built to need something new (coin label, reversed word order, Yoruba), which predicts E_b over K higher on each pair's first side than on its second by more than 0.01, both sides above their matched randoms (about 0.01)
rival_b: On real text every fine-tune writes mostly new directions through the variables it needs, whichever side, which predicts E_b over K on all six sides within the matched randoms' range or within 0.01 of each other, so E_b over K does not detect reuse against new on these realistic fine-tunes
rung: parameter
smallest_gap: 0.01
spread: 0.004
n: 3
design: unpaired
unit: seeds per task side (E211's saved fine-tunes at the scored size)
spread_source: experiments/kstar_needed/results/summary.json
spread_of: this reading (E_b over K: median seed sd of E268's 15 trained arms with K non-empty, 0.004; range 0.001-0.07)
floor: none (descriptive); the matched randoms are read beside: rival a expects every first side above them, rival b every side inside their range (E268's randoms 0.0089-0.0130)
stop: E_b over K on all six sides within the matched randoms' range: the paper says E_b over K does not detect reuse against new on these real-text fine-tunes, and no further real-text E_b run is proposed
sketch: experiments/realtext_eb/figures/sketch.png
rehearsal: 'E### rehearse:' on the MATS cluster debug QoS (one L40, $0), full read sizes (512 x 35 and 500 x 64 windows, 256 pairs, every variable): stage 0 (do-nothing E_b over K exactly 1, identity interchange zero flips, fp32 floor) then L1_s0 and T1_s0 read in full; measures the time per label and per text item, peak GPU memory, and the identity floor; these two items count toward the run
descriptive: user via writeup-lead, 2026-11-22: notify the whip-lead to get on whatever other runs we need; aim (a): if we can detect reuse/new on realistic fine-tunes

Plan: reports/plans/2026-11-23_plan_realtext-eb-over-k.md. Code: experiments/realtext_eb/run.py, plot.py; predictions
results/predictions.json; rehearsal cluster/realtext_eb.sbatch.
