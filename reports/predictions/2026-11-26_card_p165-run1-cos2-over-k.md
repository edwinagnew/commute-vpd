cell: P1.65
rival_a: The graded cos^2 clause is near 1 on every changed neuron, so the high reading does not depend on the set, and this run predicts the mean over the necessary set K at 0.999 or more on all 20 E196 pair fine-tunes, as over the KL-closure set
rival_b: The necessary set picks out the few neurons whose mechanism the fine-tune turned, so this run predicts the mean over K below 0.999 on at least one fine-tune (toward its lowest changed-part value, 0.88-0.98)
rung: parameter
smallest_gap: 0.001
spread: 0.00006
n: 2
design: paired
unit: seeds
spread_source: experiments/rho_saved_items/results/summary.json
spread_of: proxy: rho_main_replace over the KL-closure set, the same cos^2 per-part score averaged over a different set, so units match
floor: none
stop: none; the cell closes when the K value sits beside the K* value in the paper's three sentences
sketch: pending l40-worker refuses job steps (srun socket timeout 21:1x), cpu-run cannot draw it; script in scratchpad sketch_p165.py, drawn when the worker returns
rehearsal: a short rehearse pod (cluster worker down), or cluster debug if it returns, two of the 20 items at the full read size (12,316 neurons x 256 swap pairs, 512 windows), measuring seconds per item and peak memory before any pod
descriptive: user 2026-11-26 ~20:18 in whip-lead's chat: "if its missing data, run it yourself for anything indivudally under $100 and/or 6 hours"; ~21:05 "yes, open all four"; a readthrough fill reported as a description, no claim at the bar
