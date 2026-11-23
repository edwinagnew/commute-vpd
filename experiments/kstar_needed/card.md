cell: P1.54
rival_a: On these items the mean-ablation circuit K* and the needed set K (single-variable interchange on D) pick out much the same changed variables, which predicts E_b over K within the seed spread of E_b over K* on every item, so the built-edit order (gain edits above new and random) and the trained readings (RL 5-10x SFT) stand
rival_b: Mean ablation and interchange disagree on which variables matter (K* misses variables needed one at a time, or holds ones no single swap moves), which predicts E_b over K moving outside the seed spread on some items, possibly reordering the built edits or the RL against SFT gap
rung: premise
smallest_gap: 0.1
spread: 0.04
n: 3
design: paired
unit: seeds per trained arm (the same saved state read over K* and over K)
spread_source: experiments/rb_states/results/summary.json
spread_of: this reading (E_b over K*, seed sd of the E220 rl arm: 0.16, 0.14, 0.22)
floor: none
stop: E_b over K agrees with E_b over K* within the spread on every item: the paper states the write share as computed over K, with K* as its saved approximation, and no further K work is proposed
sketch: experiments/kstar_needed/figures/sketch.png
rehearsal: cluster only: every job runs on the MATS cluster debug QoS ($0, saved states; the task says "cluster debug, $0"), stage 0 (identity interchange gives zero flips; do-nothing E_b over K is 1) runs first in each job
descriptive: user, 2026-11-22, task file: "Report K vs K*, whether the built-edit order and trained readings stand"

The user's task (2026-11-22), verbatim: "New cell P1.54 (due 11-23, the experiment cutoff): E_b over the needed set K
under the user's frozen definition (theory/2026-11-22_definition_frozen.md). Your P1.53 admissibility page found E_b
scores K* from mean ablation, not K. Compute K by single-variable interchange on D for the built edits and the trained
SFT/RL states from saved files (cluster debug, $0), then E_b over K. Report K vs K*, whether the built-edit order and
trained readings stand, update the admissibility page, note writeup-lead. Start at once; results needed tomorrow."
