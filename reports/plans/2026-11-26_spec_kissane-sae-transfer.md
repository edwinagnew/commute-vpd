# Method spec: Kissane et al. 2024, base-to-chat SAE transfer (baseline (2) of E314)

Source: Kissane, Krzyzanowski, Conmy, Nanda, "SAEs (usually) Transfer Between Base and Chat Models", Alignment Forum /
LessWrong, 2024-07-18. Post: https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models
(same text at https://www.alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models).
Appendix (Qwen, Gemma tables): https://docs.google.com/document/d/1VfghHtnhQDczj0Q4-whJmnWXUMgMJZryIqjVXoeO_TY.
Code: https://github.com/ckkissane/sae-transfer (`utils.py`, `sae_transfer_script.py`). No copy in `~/commute-data/lit_review_pdfs`.
Plan using it: `reports/plans/2026-11-26_plan_e314-sae-proxies.md`.

## What they measured (verbatim)

- **Models.** Executive Summary: "We train SAEs on base / chat model pairs and find that SAEs trained on the base model
  transfer surprisingly well to reconstructing chat activations (and vice versa) on Mistral-7B and Qwen 1.5 0.5B." ...
  "they don't transfer on Gemma v1 2B".
- **Sites and layers.** Section "Investigating SAE Transfer between base and chat models": "All SAEs are trained on the
  pile on a middle layer of the residual stream. We used SAELens for training, and closely followed the setup from
  Conerly et al." Table headers: "Models: Mistral-7B / Mistral-7B Instruct. Site: resid_pre layer 16. SAE widths:
  131027"; appendix: "Models: Qwen1.5 0.5B, Qwen1.5 0.5B chat / Site: resid_pre layer 13 / SAE widths: 32768";
  "Models: Gemma v1 2B, Gemma- v1 2B It / Site: resid_pre layer 9 / SAE widths: 32768". Limitations: "in each case we
  trained SAEs on the middle layer residual stream. It's possible that the results vary across different layers and sites."
- **Metrics.** Same section: "We evaluate the base SAEs on chat activations and vice versa, using standard metrics like
  L0 norm, CE loss recovered, MSE, and explained variance." Columns: "L0 | CE Loss rec % | Clean CE Loss | SAE CE Loss |
  CE Delta | 0 Abl. CE loss | Explained Variance % | MSE". Code (`utils.py`, `get_explained_variance_from_batch`):
  `total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)`; `explained_variance = 1 - per_token_l2_loss /
  total_variance`, then averaged over tokens and batches. CE recovered (`sae_transfer_script.py`):
  `score = (zero_abl_loss - recons_loss) / div_val` with `div_val = zero_abl_loss - loss`, the SAE spliced at one site.
  Activations are rescaled per model by its own `norm_scaling_factor` ("each activation is scaled to have average norm
  sqrt(d_model)", section "Identifying failures").
- **Data.** Same section: "All evals in this section are on 50k-100k tokens from the pile. We don't apply any special
  instruction formatting for the chat models". Section "Investigating SAE Transfer on Instruction Formatted Data": "We
  take ~50 instructions from alpaca and generate rollouts with Mistral-7B-instruct using greedy sampling ... separately
  on the rollout and user prompt tokens."
- **Outliers.** "we identify activations with norm above a threshold, and exclude these from the evals" ("<1% of
  activations", "mostly stem from special tokens like BOS"); not excluded in the fine-tuned-SAE table.
- **References compared against.** Each SAE on its own model (base SAE on base, chat SAE on chat) beside the
  cross-model pair; zero ablation as the CE floor; a base SAE fine-tuned on "5 million chat activations" vs a chat SAE
  "trained from scratch (on 800M tokens)". No random or size-matched update. Their transfer cost in explained variance:
  Mistral base SAE 68.1% on base vs 60.9% on chat; Qwen 74.4% vs 59.2% (FVU gaps about 0.07 and 0.15).

## Departures for E314

| # | theirs | ours | why | what it could change in the reading |
|---|---|---|---|---|
| 1 | base vs chat model (full instruction tuning, 0.5B-7B) | 67M 4-layer model, 48 small known-answer fine-tunes (16 tasks x 3 seeds) | the paper's question is these updates | gaps far smaller than theirs; the scale of their 0.07-0.15 is no guide to ours |
| 2 | SAEs on both models (base and chat), each tested on the other | base-model SAEs only (E234) | the baseline asks whether a base dictionary sees the update; no per-state SAE training | no "own SAE" reference, so a gap cannot be split into drift vs a harder-to-reconstruct distribution |
| 3 | L1 SAEs (SAELens, Conerly et al.), width 32k-131k | BatchTopK, SAEBench recipe, width 16,384, k 80, 500M Pile tokens | E234's SAEs exist and follow a current standard | L0 is fixed at k under TopK, so their L0 shift on transfer has no analogue; TopK may hide drift that L1 shows as more active latents |
| 4 | one middle-layer resid_pre site | 13 sites (resid_pre0; attn, mlp, resid_post of 4 layers), mean over sites | a small update may land in one block; block outputs are where E314's error term reads | the mean dilutes a one-site change; per-site values are saved for that |
| 5 | explained variance as a per-token ratio, then averaged (code above) | FVU = ratio of sums over all positions (one minus explained variance, but pooled before dividing) | ratio of sums is stable at low-variance positions | absolute FVU differs from 1 - their explained variance; the sign and ordering of gaps should not |
| 6 | each model's activations rescaled by its own norm factor | base normaliser folded into the SAE, fine-tuned activations fed unscaled | the fine-tuned model shares the base's scale up to a small update | a norm change shows up in our gap; theirs removes a global norm shift first |
| 7 | 50k-100k Pile tokens; ~50 Alpaca rollouts | all positions of each task's own rows | the gap must be read where the task update acts | task rows are narrow, so FVU and its noise floor differ from Pile values |
| 8 | outlier-norm activations (BOS, early positions) excluded | none excluded | 67M has no measured outlier problem; exclusion would need its own threshold | a few high-norm positions could dominate the ratio of sums; check per-position error on the first state |
| 9 | no update-size reference | a matched-size random update per state (Gaussian per matrix at that matrix's Frobenius norm), our adaptation | without it a gap cannot be told from update size | reading is "trained vs random at equal size", a claim they do not make |

## Their metrics we do not compute

- CE loss recovered (SAE spliced in, zero ablation floor): not computed; E314 reads log-odds of the target instead.
- Clean / SAE / zero-ablation CE loss and CE delta: not computed, same reason.
- L0: not read; fixed at k = 80 by BatchTopK.
- MSE (unnormalised): not reported; FVU is its normalised form.
- Outlier-norm fraction and token breakdown: not computed.
- Rollout vs user-prompt split, and the fine-tuned base SAE: not run.
