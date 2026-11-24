cell: P2.68
rival_a: whole-group head CMAP tracks the definition's reuse label (a reuse fine-tune's gain is carried by base-model heads that the fine-tune re-weights, so swapping the fine-tuned heads' outputs into W0 recovers more of it than for a new fine-tune) predicts the reuse side's whole-circuit CMAP log-odds share exceeds the new side's by more than 0.07 on most of E187's nine built pairs (within each format)
rival_b: CMAP reads activation sufficiency at the read, which any trained fine-tune routed through attention has whether it reuses or builds (Prakash et al. established base-model use separately) predicts reuse-minus-new differences within 0.07 of zero, or of either sign, across the nine pairs
rung: strategy
smallest_gap: 0.07
spread: 0.0314
n: 9
design: paired
unit: E187's built pairs (one reuse and one new fine-tune at matched size, both read once on the definition's rows)
spread_source: experiments/share_stability/results/run.json
spread_of: proxy: E161's batch sd of the re-gating log-odds share over five 64-row batches of one fine-tune (key .stage1_step10.sd_r = 0.0314); same unit (log-odds share of the gain, mean numerator over mean gain); stage 0 measures this run's own per-task gate N, L from 15 random updates
floor: per-task gate from stage 0: Δcorrect > N (at least 1 when N = 0) and gain G > max(L, 1 log-odds); rival a expects every pair's two sides to clear it (built pairs were sized to gains over 1), rival b the same, so the floor separates neither
stop: the reuse-minus-new difference sits within 0.07 on the nine pairs: the paper reports whole-group head CMAP as a baseline that does not recover the built labels, and no further CMAP variant runs without a new cell
sketch: experiments/cmap/figures/sketch.png
rehearsal: cluster only: stage 0 (every positive control, floors, one timed item) runs alone on a debug L40 with exit-1 stops; stage 1 shards on debug; a pod only if the timed estimate exceeds 8 debug shards, capped at $16 terminate / $20 total
