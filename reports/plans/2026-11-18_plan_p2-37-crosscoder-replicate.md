# Plan: our crosscoder pipeline against the authors' released numbers (cell P2.37, E210)

We load the crosscoder Minder et al. 2025 trained on Gemma-2-2B and its chat model, and run our own code on it: the
relative decoder norm and the latent-scaling ratios. Their released table gives their values for every latent. If ours
match theirs, our port is right and E203's crosscoder readings can be cited; if not, we find where it differs before
any crosscoder is rerun.

Card: `reports/plans/2026-11-18_plan_p2-37-card.md`. Rules and expected values: `experiments/crosscoder_replicate/results/predictions.json`.
Code: `experiments/crosscoder_replicate/run.py` (stages `a`, `b`), tests `tests/test_crosscoder_replicate.py`.

## What runs

| Stage | Where | What | Expected | Stop |
|---|---|---|---|---|
| a | CPU, `cluster/crosscoder_replicate_a.sbatch` (16G, 20 min) | download the crosscoder (2.72 GB) and `feature_df.csv` (41.5 MB); read the model order off the decoder norms; our relative norm against `dec_norm_diff` on all 73,728 latents | max abs diff ~1e-7 (CSV rounding: their own norm columns reproduce `dec_norm_diff` to 5.7e-8); top-3,176 sets identical | diff > 1e-5 |
| b | L40, debug QoS, `cluster/crosscoder_replicate_b.sbatch` (40G host, 30 min) | 1M FineWeb + 1M LMSYS tokens through gemma-2-2b and gemma-2-2b-it, output of `model.layers[13]`; the crosscoder's FVE (control, first); our betas on the top 3,176 latents by relative norm, with BOS kept (primary) and dropped (reported) | FVE 0.83-0.90 each model (paper Table 3: 86.2%, 86.9%); Spearman of nu_r and nu_eps about 0.95; pass count 570-630 against their 599 | FVE < 0.7 on non-BOS tokens |

**Reproduced** (filed): Spearman >= 0.9 for nu_r and for nu_eps, on latents both sides keep, and our pass count
(nu_r < 0.5 and nu_eps < 0.2) within 10% of feature_df's 599. Why these bars: adding noise of sd 0.05 to
feature_df's own ratios keeps Spearman at 0.96-0.97 and moves the count by 4%; a 5% systematic scale moves it by 7%.
Relative norm alone (the do-nothing reference) reaches only 0.51 and 0.67, so the bar separates a working port from
one that only ranks by norm. 2M tokens instead of the paper's 50M: per-latent beta noise grows about 5x; the median
top latent still fires ~1,000 times on the LMSYS half, 477 fire under 100 times and ~3 under once (from
feature_df's `lmsys_freq`), so we expect Spearman ~0.95 and under 10 nan latents.

## Data and settings (all sourced in predictions.json)

- Crosscoder `science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss` (BatchTopK, k 100, 73,728
  latents, no activation normaliser in the file, inference threshold stored). `lm_name` lists the chat model first;
  the code reads the order from the decoder norms and flips the file so index 0 is the base.
- Tokens: the authors' copies `science-of-finetuning/fineweb-1m-sample` and `.../lmsys-chat-1m-chat-formatted`
  (its `text` column is already in the Gemma chat template), train split, first rows, 1,024 tokens per document at
  most with BOS, as the toolkit's preprocessing; 1M tokens a set.
- Models in bf16 with eager attention; activations kept in bf16 (as the toolkit stores them), betas in fp32.

## Cost

$0 (cluster). Stage a: ~5 min (2.7 GB download at ~50-100 MB/s, ~8 GB host RAM). Stage b, estimated, not measured:
model downloads ~10 min worst case (two checkpoints of up to ~10 GB each), forward passes through 14 of 26 layers
for ~4,300 documents a model ~2-4 min, FVE and two beta passes ~3-5 min: ~15 min of the 30-min limit. GPU memory:
activations 2M x 2 x 2304 bf16 = 18.4 GB, one Gemma at a time 5.2 GB, crosscoder 2.7 GB, encode working set ~5 GB
per 4,096-token chunk: peak ~31 GB of 48. Host: ~8 GB crosscoder load + ~10 GB model load, 40G requested.
Needs `HF_TOKEN` (Gemma-2 licence) in the submitting shell; the two datasets are ungated copies.

## Departures and recorded discrepancies

- Ours has no negative-beta filter (paper.txt 2272); this check applies the paper's filter to both sides and records
  the departure, not fixed here. feature_df holds 306 negative base error betas and 54 negative base reconstruction
  betas among the top 3,176, while the paper says "None in BatchTopK": recorded as the authors' own inconsistency.
- Paper Table 1's Gemma counts (301, 979, 2035, 3269 at 0.2-0.8) match feature_df's unfiltered counts only at 0.4
  (979); both filtered and unfiltered counts are reported.
- 2M latent-scaling tokens, not 50M (the cell's sizing); first rows, not their sample.
- BOS: kept (toolkit setting for Gemma-2); the BOS-dropped reading is reported to locate a mismatch from it.

## Not run here

The crosscoder's training (we use their weights); latent scaling on base-only latents; the causal and
autointerpretability checks of the paper; any rerun of E203.
