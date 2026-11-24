cell: P2.77
rival_a: Seed count and variant pooling were cosmetic, so the standardised figure predicts every task's seed mean within its old seed range on every reading (share, steering, erank, prior, r_n, top-neuron k64, k_half, size)
rival_b: The old pooled variants or extra seeds shifted some readings, so the standardised figure predicts at least one task mean moving outside its old seed range (e.g. e46 or e53 once the b / broad variant is dropped, or e46 top-neuron share once kmatched becomes k64)
rung: parameter
smallest_gap: 0.1
spread: 0.02
n: 3
design: unpaired
unit: seeds
spread_source: experiments/method_table/results/final.json
spread_of: this reading (re-gating share, e53 cipher new side, 4 states)
floor: none
stop: none; the cell closes when Fig 4 and known_appendix are replotted from 3 seeds a side
sketch: reports/predictions/2026-11-24_sketch_p277-known-3seeds.png
rehearsal: cluster only: five 67M fine-tunes and their readings, each a few minutes on a cluster debug GPU, $0; no pod
descriptive: user 2026-11-24 in whip-lead's chat: "yes rerun whatever makes fig 2 consistent. same for all figs in the results section. experiemental rigour is essential!!!!"; a consistency rerun reported as a description, no claim at the bar
