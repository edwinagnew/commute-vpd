cell: P2.70
rival_a: the plant latents' negative base betas are sampling noise of a true zero at our 100,000-token fit predicts their bootstrap share below zero has a median between 0.5 and 0.95, their 50M base beta shrinks to under a fifth of the 100,000-token value (median), and fits A and B agree negative on at most 5 of 10
rival_b: the plant latents' negative base betas are real, so Minder et al.'s exclusion stands on them predicts a bootstrap share over 0.95 (median), a 50M base beta over half its 100,000-token value (median), and at least 9 of 10 latents negative in both halves and below zero by more than 2 standard errors
rung: parameter
smallest_gap: 0.3
spread: 0.144
n: 10
design: paired
unit: plant latents (dropped by the literal rule at 100,000 tokens, both ratios above -0.02, carrying at least 0.01 of the change), pooled over the ten planted controls, each read at both fits
spread_source: experiments/crosscoder_proper/results/refit/predictions.json
spread_of: proxy: the bootstrap share of one noise latent, uniform on 0.5 to 1 (sd 0.5/sqrt 12), derived not measured; stage 0 measures the real per-latent spread before stage 1 spends
floor: per latent, redteam-lead's filed rule: an exclusion counts as real only when both 25M halves are negative and the 50M fit is below 0 by more than 2 bootstrap standard errors; rival_a expects at most 2 of 10 real, rival_b at least 9 of 10
stop: stage 0 misses its control (the rebuilt side's energy off by over 1e-4, or the saved betas not reproduced within 3 jitter floors): the refit is not run and the 100,000-token reading stands with the label "not separable from latent-scaling noise"
sketch: experiments/crosscoder_proper/figures/refit_sketch.png
rehearsal: E### rehearse: stage 0 itself on the pod, pod commute-E###-rehearse-betas (cluster l40-worker /ephemeral is full): the ten controls' first chunk at full size, the reproduction check against the saved betas, the jitter floor and the block bootstrap; it measures the seconds per chunk, the fit's seconds per chunk, peak GPU memory and the per-latent bootstrap spread, and stage 1 runs on the same pod only after it passes
