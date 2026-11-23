cell: P2.65
rival_a: the cheap readings follow training history (an unlock re-weights a circuit the base already has) predicts the diagonal ranks above both off-diagonals in at least 7 of 9 task-seed triples for steering share at matched update size, AUC at least 0.85, with the matched Gaussians at AUC near 0.5
rival_b: the cheap readings follow only update size and output change, not history predicts at matched update size about 3 of 9 triples (chance) and AUC near 0.5 for steering, neuron and erank, real and Gaussian alike, while at matched gain size and Pile KL separate trivially
rung: premise
smallest_gap: 0.1
spread: 0.032
n: 9
design: unpaired
unit: fine-tuned states (task x seed), 9 diagonal against 18 off-diagonal
spread_source: experiments/procedural_rl/results/read/brackets_sft_s0.json
spread_of: proxy: E256's steering share (steering.share.lo) over 3 SFT seeds on one procedural task (brackets 0.334, 0.270, 0.306; the largest of E256's three tasks), the same reader in the same unit on the same 67M
floor: MIN_GAIN 1 log-odds (a share is nan under it; G* = half the installed cued gain is set at or over 1 by the filed cued-gain bars); matched Gaussians must gain under 1 and their AUC under 0.85; rival_a expects Gaussian AUC ~0.5, rival_b the same, so the floor separates neither by itself
stop: controls (2) or (3) miss (the diagonal is not 5x faster, or its update transplants onto M): reuse by training history is then not built, the readings are not run, and the line closes with that finding
sketch: experiments/installed_cross/figures/sketch.png
rehearsal: E271 rehearse: the whole controls stage at full size on cluster debug L40 ($0): ctl, install x3, cross x4, controls, each job <= 30 min, at the full read sizes (256 stop + 256 read rows per task, 27 + 9 fine-tunes); it measures step time per install and fine-tune step, steps to G*, peak GPU and host memory, and the per-state read time in a first read job; no pod is used for the controls or the decomposition-free readings
