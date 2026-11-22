cell: P2.55
rival_a: the key copy and the low-prior copy run through M's copy heads, and one head was too few to show it; this predicts that with M's top 3 heads by prefix score switched off together each copy loses 0.5-0.8 of its gain (E62's copy about 0.8), while both ciphers, rank1 and elicit stay under 0.05 at every k and k random heads (outside the top 5 and the previous-token head (1,1)) stay under 0.1
rival_b: most of each copy runs through heads that are not ranked high on M's copying, so (2,4)'s 0.25 was most of what the copy heads carry; this predicts that key copy still loses under 0.4 of its gain with the top 5 heads off, while E62's copy rises well over its 0.53
rung: parameter
smallest_gap: 0.15
spread: 0.041
n: 3
design: paired
unit: fine-tune seeds 0-2 per task; the k arms and the null are read on the same states and rows
spread_source: experiments/prior_mech_2x2/results/stage1/e58-copy_s0.json
spread_of: proxy: the k = 1 share, the (2,4) share of runs 1-2, sd over seeds of key copy (0.245, 0.291, 0.209 in results/stage1/e58-copy_s{0,1,2}.json; task mean 0.248 in results/stage1/summary.json); the low copy's is 0.034 (results/stage2/head/); the k-head share is the same quantity with more heads off
floor: G_min, run 1's gain gate of 5.15 log-odds: a state under it is not read. Every key and low-copy state gains 9-20 (results/stage1, stage2/head), over it under both rivals; the random updates gain about 0.03, under it under both rivals, so they are not read
stop: key copy loses under 0.4 of its gain with M's top 5 copy heads off: the copy does not run through the heads ranked high on M's copying, and the 2x2's reused corners cannot be labelled by switching off copy heads
sketch: /mnt/nw/home/e.agnew/commute-data/scratch/compare-lead/khead/sketch.png
rehearsal: cluster only: ~9 min on the MATS debug L40 at $0, no pod; nothing is trained, and its two controls (M's copying with (2,4) off repeats E62's 0.0095; the k = 1 shares repeat runs 1-2's saved (2,4) shares within twice a floor measured in the same job) run first and exit on a miss
