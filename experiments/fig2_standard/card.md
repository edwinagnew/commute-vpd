cell: P1.62
rival_a: The old Fig 2's differences between columns came from the set-up (SAE read over K* instead of the needed set K, SAE on the scaled states, VPD on seed 0 only, different randoms and remainder conventions), so this run predicts at least one block's per-task values move by more than 0.05 at one featurization when all three read the same states, K rule, randoms and convention
rival_b: The set-up differences did not matter, so this run predicts every featurization's per-state values stay within 0.05 of the old figure's values for the states both read, and the old qualitative order (built reuse above built new and randoms at every featurization; trained near its random) holds
rung: parameter
smallest_gap: 0.05
spread: 0.003
n: 5
design: paired
unit: key-grid tasks (copy, elicit, rank1, prior_cipher, cipher), each the mean of 3 seeds, read against its own size-matched random; plus 5 built edits each against its own size-matched random; descriptive, no bar
spread_source: experiments/ia3_control/results/summary.json
spread_of: proxy: neuron E_b_K of E290's full fine-tunes, sd over 3 seeds within a task (0.000-0.004), the same reading at one featurization
floor: none
stop: Any stage-0 control at any featurization reads other than expected (do-nothing E_b 1; identity swap 0 flips; a saved E268 or E290 item re-read through this run's driver within its measured floor of the saved value): the job stops and the cell waits on the fix
sketch: experiments/fig2_standard/figures/sketch.png
rehearsal: E298 rehearse: on a cluster debug L40: stage 0 (do-nothing, identity swap, one saved-item reproduction) at each featurization at the full read sizes (256 pairs, the item's own windows and positions, the full SAE dictionaries), plus one trained item per featurization timed; measures seconds per item, peak GPU memory, and the reproduction floor
descriptive: user, 2026-11-24, in whip-lead's chat: "yes rerun whatever makes fig 2 consistent. same for all figs in the results section. experiemental rigour is essential!!!!"; reported as a description, no claim at a bar

## What runs (run 1 of cell P1.62)

One driver reads E_b over the needed set K (single-variable interchange, 256 pairs = 4 derangements x 64 inputs, the
state's own windows and read positions) at VPD, top neurons and SAE latents, on the same 42 states: 7 built edits, 5
size-matched randoms for the 5 non-random built edits, E147's 15 full fine-tunes (5 tasks x 3 seeds, unscaled) and
E290's 15 matched randoms. Every featurization's remainder or error variables are in the scan and in E_b (headline);
E_b without them is reported beside it. Saved reads that already match this rule exactly are reused after a
reproduction check; the rest run. Plan: reports/plans/2026-11-24_plan_p1-62-fig2-standard.md.
