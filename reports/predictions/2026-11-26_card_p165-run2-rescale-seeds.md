cell: P1.65
rival_a: VPD and SAE readings over K cannot see a gain-only update as kept, so this run predicts the rescale-only control at seeds 1-2 (VPD) and 0-2 (SAE) within the full fine-tunes' range (VPD 0.027-0.042, SAE 0.031-0.047)
rival_b: Seed 0 at VPD was low by chance, so this run predicts seeds 1-2 at VPD, or the SAE seeds, reading well above the full fine-tunes (0.2 or more)
rung: parameter
smallest_gap: 0.05
spread: 0.002
n: 3
design: paired
unit: seeds
spread_source: experiments/fig2_standard/results/definition_rows.csv
spread_of: proxy: E_b_K of the full fine-tunes across seeds at VPD, the same reading on the same tasks, so units match
floor: none
stop: none; the cell closes when the per-seed values sit beside the full fine-tunes' values
sketch: pending l40-worker refuses job steps (srun socket timeout 21:1x), cpu-run cannot draw it; script in scratchpad sketch_p165.py, drawn when the worker returns
rehearsal: a short rehearse pod (cluster worker down), or cluster debug if it returns, E298's stage 0 (do-nothing, identity, must-flip) plus one gain item at VPD and SAE at full read sizes, measuring seconds per item before any pod
descriptive: user 2026-11-26 ~20:18 in whip-lead's chat: "if its missing data, run it yourself for anything indivudally under $100 and/or 6 hours"; ~21:05 "yes, open all four"; a readthrough fill reported as a description, no claim at the bar
