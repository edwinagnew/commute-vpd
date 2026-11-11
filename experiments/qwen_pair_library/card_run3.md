cell: P3.5
rival_a: the built pairs are what they were built as; at a recipe that learns both sides (peak 1e-5, 300 steps), the reuse side generalises to held-out keys and the new side does not; this predicts at the last save copy held-out exact match near 1.0 and cipher's more than 0.3 under copy's, with cipher's trained exact match at or above 0.9
rival_b: the cipher's letter map is learned as a general rule once trained long enough, so both sides generalise; this predicts cipher held-out exact match at the last save within 0.3 of copy's
rung: premise
smallest_gap: 0.3
spread: 0.2557
n: 256
design: paired
unit: rows (held-out keys; held-out exact match difference copy - cipher at the last save, same keys both sides, per seed)
spread_source: experiments/qwen_pair_library/results/rehearse_copy.json
stop: a new side still does not learn on trained items (exact match under 0.9 at step 300) on all three seeds: that pair is dropped and the library is the twins plus whichever built pair passed
sketch: experiments/qwen_pair_library/figures/sketch_run2.png
