cell: P2.55
rival_a: the re-gating share follows the mechanism (a fine-tune that runs through the copy heads reads high whatever the base prior) predicts the cued copy, a copy map behind a new cue whose answer the base gives about 10^-5.9, reads r near key copy's 0.77 and well above cipher's 0.27, so r's mechanism effect is the larger of its two effects across the four corners
rival_b: the re-gating share follows the base prior (r tracks how likely the base already found the answer) predicts the cued copy reads r near cipher's 0.27 although it runs through the copy heads, so r's prior effect is the larger
rung: premise
smallest_gap: 0.2
spread: 0.012
n: 3
design: unpaired
unit: seeds
spread_source: experiments/key_grid/results/stage1.json
spread_of: this reading: r (log-odds form) at E237's common update size, sd over seeds 0-2 of the cipher task (0.276, 0.259, 0.281), the widest of the four corner tasks E237 read
floor: the head label: the cued copy must lose at least 0.2 of its gain with (2,4) off (each seed over its median random head) and not under 0.2 with the top 3 copy heads off, and key copy and both ciphers keep run 1's labels, or no corner is labelled and nothing is called; both rivals expect the cued copy over the floor (E62's copy map, the same form, lost 0.47-0.56 to (2,4))
stop: no filed B band clears the construction check (read prior at or under 10^-5.6 with at least 1000 train and 250 read rows on 24 keys, every read key with 8 train windows), or the cued copy loses under 0.2 of its gain with the top 3 copy heads off: a low-prior reused corner cannot be built as its own task this way, and the 2x2 stays with the row-selected low copy and its caveat
sketch: /mnt/nw/home/e.agnew/commute-data/scratch/compare-lead/cued-copy/sketch.png
rehearsal: cluster only: the run is ~7 min on the MATS debug L40 at $0 (three band selections, three 300-step fine-tunes of ~11 s, reads of 6 items), no pod; run 1's control, the repeats of E133's prior and E147's shares, and the construction check run first and exit on a miss, before any training
