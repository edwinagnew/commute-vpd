cell: P1.57
rival_a: The write share tracks reuse on these older items as it did on the built pairs, which predicts E_b over K for E62's copy fine-tunes at least 0.2 above its lookup fine-tunes, and the P2.50 key-grid tasks ordered as their squared-cosine R panels (copy highest, cipher lowest), each above its matched random
rival_b: E62's copy and lookup both mostly write new directions at the marker (the copy share 0.47-0.56 was partial), which predicts E_b over K for copy within 0.2 of lookup, and the key-grid tasks within 0.2 of one another and of their matched randoms
rung: strategy
smallest_gap: 0.2
spread: 0.04
n: 3
design: paired
unit: seeds per arm (E62 copy and lookup, seeds 0-2; P2.50 tasks, seeds 0-2)
spread_source: experiments/rb_states/results/summary.json
spread_of: proxy: E_b over K* at VPD, seed sd of the E220 rl arm (0.16, 0.14, 0.22), the same write share on other trained states
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23); both rivals expect |K| of hundreds or more
stop: If E62's copy reads within 0.2 of lookup over K, the worked example is reported as one the write share does not separate, and no further E62 reads are proposed
sketch: experiments/eb_dropped/figures/sketch.png
rehearsal: E278 rehearse: on a cluster debug L40 ($0): stage 0 (identity swap zero flips; do-nothing E_b over K is 1) and one E62 state at full read sizes, measuring time per item; scored items on cluster debug if it starts at once, else H100 pods (user 2026-11-23: "if it would be quicker to use runpod, do so")
descriptive: whip-lead 2026-11-23 (drop audit the user asked for): "neither was ever read under the frozen definition"; result to redteam-lead for a verdict

## What runs (E278, cell P1.57)

E_b over the needed set K with E268's apparatus at VPD (as E275 extended it to new items) on: stage 0 first; E62's
copy and lookup fine-tunes (seeds 0-2, the user's worked example); then E237's P2.50 grid (five 96-key tasks scaled to
copy's update size, seeds 0-2, each with its saved Gaussian random at matched per-matrix norms), skipping any state E268
or E275 already read. |K|, F's top-1 minus top-2 margin and E_b over K beside every item; |K| < 100 reads "too small to
read". The key-grid figure is redrawn with E_b over K.
