cell: P3.13
rival_a: The library's reuse sides (copy, known facts, gold maths) change mostly what Qwen's own MLP neurons and attention heads already write, as on the 67M, which predicts the held-model reading S of each pair's reuse side above its new side (cipher, invented facts, shuffled gold) by more than 0.005 at every seed, at both saves, at saved size and with the larger side shrunk to the smaller side's size, on all three pairs
rival_b: At 1.7B, with 16,384 fit positions for a 6,144-wide site, the least-squares map from M's units fits either side about equally well, or a gap comes only from the sides' sizes, which predicts no pair separating: on each pair some save, at saved size or size-matched, has a seed gap within 0.005 or gaps of both signs
rung: parameter
smallest_gap: 0.005
spread: 0.0041
n: 3
design: paired
unit: seeds (0, 1, 2) of each library side, paired by seed number as P3.9 paired them; one gap per seed, S(reuse side) minus S(new side), in S's own units (1 = F's output fully kept by H)
spread_source: experiments/qwen_held_readings/results/summary.json
spread_of: this reading: run 1's seed sd of S on the E167 step-75 states, the largest of its three runs (own-samples SFT, 0.004095, n 2), rounded up; the same code on Qwen, other rows. Run 2 reports each side's own seed sd (n 3) beside it
floor: S reads 'no change' when KL(F||M) on the read rows is at or under 100 times the row set's do-nothing KL(F||H), measured in the same job before that row set's first state; both rivals expect every library state above it (run 1's floor 3.5e-7 on maths rows; the library's states moved their own rows far more). A Gaussian change at the smallest states' norms may read 'no change' near it; the rule does not read the random changes
stop: A control misses: F = M not read as KL(F||M) at most 1e-9 with S 'no change' on a row set, or the pod's first state's o_proj and down_proj change not read as S at least 0.999. The pod stops before any further state and the port is checked first. This is the cell's last run: section 6 is written from its outcome as read, and no further S run on the library follows without a new cell from the user
sketch: experiments/qwen_held_readings/figures/sketch2.png
rehearsal: E??? rehearse: cluster debug L40, cluster/qwen_held_run2.sbatch --pair copy_cipher --limit 1: the two controls, then the pod's first state (cipher seed 0, late save) fetched from W&B, its saved-size read, its rescaled read (x 0.458) and both random changes, at the full read sizes (128 rows of 256 tokens, 64 fit, 64 read; Qwen3-1.7B, all 56 output matrices); measures time per S read on an L40, the fetch speed from W&B, peak GPU (Grams held on the host so it fits 44 GiB) and host RAM, and the library rows' do-nothing floor

## What runs (run 2 of cell P3.13, rung L3.2)

Plan `reports/plans/2026-11-19_plan_l3-2-qwen-readings.md` (Runs, item 2) and `_appendix.md`; predictions
`results/run2/predictions.json`; code `run2.py`, `summary2.py`, `plot2.py`. The held-model reading S of run 1 (the same
`commute.qwen_held` code) on the pair library: copy against cipher, known against invented facts, gold maths against
shuffled gold, seeds 0-2, at the late and matched saves P3.9 and E187 read. Each state is fitted on 64 of its arm's own
128 rows and read on the other 64, beside a Gaussian change at its own per-matrix norms; each state of the pair's larger
side is read again times E187's factor c (0.38-0.97), beside the same Gaussian times c. One H100 pod per pair; 12
states and 36 S reads a pod; controls first and alone. The rule compares the two sides' S per seed against a bar of 0.005, the
67M's smallest such gap (E197), above run 1's seed spread of 0.0041, at saved size and size-matched. No update is called
reused or new.
