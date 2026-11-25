cell: P2.80
rival_a: whole-group head CMAP tracks the built reuse label predicts the 8 reuse tasks' CMAP head share above the 8 new tasks' on most of Fig 4's pairs, on all 3 seeds
rival_b: CMAP reads activation sufficiency at the read, which any trained fine-tune has predicts reuse and new tasks' shares overlapping, of either sign across the pairs
rung: strategy
smallest_gap: 0.07
spread: 0.0314
n: 9
design: paired
unit: Fig 4's built pairs, 3 seeds a side
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the re-gating log-odds share (same unit as the CMAP log-odds share, as in E288's card)
floor: E288's per-task gate (dcorrect > N, gain > max(L, 1 log-odds)); both rivals expect every side to clear it
stop: the 5 missing seed-2 states read under E288's code and controls; no further CMAP variant without a new cell
sketch: experiments/cmap/figures/sketch_p2-80.png
rehearsal: cluster only: 5 states on a debug L40 with E288's stage 0 controls (do-nothing 0, planted head 1.0) re-run first, about 3 min per item measured in E288
descriptive: user 2026-11-24 via whip-lead ("yes, open a cell for running the model-diffing tools on the same fine-tunes"), a published baseline reported as a description
