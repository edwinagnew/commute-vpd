cell: P2.54
rival_a: the re-gating share follows the mechanism (a fine-tune that runs through the copy head reads high whatever the base prior) predicts the low-prior copy reads r near key copy's 0.77 and well above cipher's 0.27, so r's mechanism effect is the larger of its two effects across the four corners
rival_b: the re-gating share follows the base prior (r tracks how likely the base already found the answer) predicts the low-prior copy reads r near cipher's 0.27 although it runs through the copy head, so r's prior effect is the larger
rung: premise
smallest_gap: 0.2
spread: 0.012
n: 3
design: unpaired
unit: seeds
spread_source: experiments/key_grid/results/stage1.json
spread_of: this reading: r (log-odds form) at E237's common update size, sd over seeds 0-2 of the cipher task (0.276, 0.259, 0.281), the widest of the four corner tasks E237 read
floor: run 1's head label: the low-prior copy must lose at least 0.2 of its gain with (2,4) off (and more than its median random-head share), key copy must pass run 1's P1 and both ciphers its P2, or no corner is labelled and nothing is called; rival a and rival b both expect the low-prior copy over 0.2 (it is the same task on harder rows)
stop: the construction check fails (no set of copy windows 1.5 decades under key copy's prior with at least 256 train and 64 read rows on 48 keys), or the low-prior copy loses under 0.2 of its gain with (2,4) off: the low-prior reused corner cannot be built this way, and the 2x2 waits for another construction
sketch: experiments/prior_mech_2x2/figures/card2_sketch.png
rehearsal: cluster only: run 2 is ~10 min on the MATS debug L40 at $0 (selection, three 300-step fine-tunes of ~9 s each, reads of 21 items), no pod; its controls (the prior path repeats E133's copy prior, holes_fill's readers repeat E147's shares on copy_s0) and the construction check run first and exit on a miss, before any training
