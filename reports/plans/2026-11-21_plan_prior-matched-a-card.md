cell: P2.57
rival_a: the cheap readings follow the mechanism (copy runs through M's copy heads, the prior-matched cipher does not) predicts that at 10 seeds each reading still splits copy from the prior-matched cipher, exact Mann-Whitney p at or under 0.01 and the gap over the 3-seed band, with E237's signs (head +0.25, r +0.40, erank +68, steering pooled +0.57, neurons k64 +0.015; per-key steering reversed)
rival_b: the cheap readings follow the answer prior, 0.20 decades apart for this pair predicts that E237's 3-seed gaps were seed luck and at 10 seeds the rank test reads p over 0.01 or the gap falls inside the band, so the readings do not split the pair
rung: parameter
smallest_gap: 0.0067
spread: 0.0038
n: 10
design: unpaired
unit: seeds (fine-tunes of one task by E147's recipe, one per seed)
spread_source: experiments/prior_mech_2x2/results/stage2/e58-prior_cipher_s0.json
spread_of: this reading (neuron share at k matched, the hardest of the eight to resolve: its E237 band 0.0067 is the larger 3-seed range, its sd 0.0038 the prior-matched cipher's over seeds 0-2 in stage2/e58-prior_cipher_s{0,1,2}.json; every other reading's gap is 1.1 to 100 bands)
floor: holes_fill's gain floor 1 log-odds and khead's gain gate G_min 5.15: both rivals expect copy (gain ~16) and the cipher (~12) over both at every seed; a random update (gain ~0.03) and the do-nothing (0) read nan, and the do-nothing must read exactly 0 gain
stop: the labels miss at 10 seeds (copy's mean (2,4) share under 0.2 or a seed under its null median), or every reading reads does_not_separate: the matched-prior pair no longer separates reused from new and claim 2 loses its pair evidence; write that up rather than adding pairs
sketch: /mnt/nw/home/e.agnew/commute-data/scratch/compare-lead/prior-matched-writer/sketch.png
rehearsal: cluster only: run A is one debug-QoS L40 job ($0, about 27 min), no pod
