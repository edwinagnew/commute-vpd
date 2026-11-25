cell: P2.83
rival_a: E203's plant always sat at the answer position, where base latents fire, so the crosscoder wrote it into those latents' fine-tune decoders (saved decoder: difference along the plant at cosine 0.99999, base decoder orthogonal to it, error ratio = the base decoder's own overlap 0.244) predicts the content-independent plant passes the authors' rule at both crosscoder seeds (lead latent relative norm near 1, nu_eps near 0.02, specific part near 0.77 as E212's spread plant read), and E203's own placement still reads nu_eps near 0.24 at 200M tokens
rival_b: this crosscoder at block 2 of the 67M under the authors' recipe cannot isolate a fine-tune-only direction at all predicts the content-independent plant also misses (specific part under 0.5 at both crosscoder seeds)
rung: parameter
smallest_gap: 0.05
spread: 0.033
n: 8
design: unpaired
unit: tasks per built side (each the mean of 3 fine-tune seeds, one crosscoder per state)
spread_source: experiments/crosscoder/results/negbeta_rescore.json
spread_of: proxy: the crosscoder reseed of d on one state (e54/elicit_s0, crosscoder seeds 0 and 1: 0.416 and 0.463, sd 0.033), the same reading in its own units; the fine-tune-seed spread per task is read by this run
floor: the planted control: the lead latent passes the authors' test (nu_r under 0.5, nu_eps under 0.2, base betas at least 0) and fine-tune-specific latents carry at least 0.5 of the planted change; rival a expects about 0.77 for the content-independent plant, rival b under 0.5. The self control: 0 fine-tune-specific latents, relative norms within 0.001 of 0.5
stop: the content-independent plant misses at both crosscoder seeds: the crosscoder cannot find a fine-tune-only direction at block 2 of the 67M under the authors' recipe; the 48 values are plotted with that control result beside them and no further crosscoder run is proposed
sketch: reports/plans/2026-11-25_plan_p2-83-crosscoder-fix-card_sketch.png
rehearsal: 'E311 rehearse: full read sizes, tiny dict' on a cluster L40 (cluster/crosscoder_fig4_rehearse.sbatch): every stage (self three times with a stop-and-resume, the three plants, the long plant, two stage-2 shards covering a key-value state, an e53 state and a Gaussian update) with a 1,536-latent dictionary on 4.1M tokens, every read at full size (8,192 val rows, 100k latent-scaling tokens, the full score windows); measures time per item, peak GPU and host memory, the resume floor
# descriptive: as P2.80, the crosscoder's d on Fig 4's states is reported as values beside its controls with no verdict on reuse (the definition makes no call from a tool's reading; P2.80 analysis, 2026-11-24)
