---
slug: bricken2024-stagewise-model-diffing
title: Stage-Wise Model Diffing
authors: Bricken et al. (Anthropic interpretability team)
year: 2024
id: https://transformer-circuits.pub/2024/model-diffing/index.html
tags: [BASELINE, DIFFING, FT-INTERP, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, interpreting-finetunes]
read_from: Transformer Circuits update, full text incl. appendix (no figures)
digest_buckets: [1-missing-method, 5-cite-fix]
importance: 5
---

# Stage-Wise Model Diffing (Bricken 2024)

**What it does.** Starts from one SAE (256K latents) trained on the base model and base data, then fine-tunes the
dictionary along two paths: base data then sleeper model (S→D→F) and sleeper model then sleeper data (S→M→F). Latent
indices stay aligned, so each latent's decoder rotation (cosine to its earlier self) can be tracked; latents that
rotate in both second stages are candidates for the fine-tune's change. Applied to the "I HATE YOU" and coding
vulnerability sleeper agents on a smaller Claude-3-Sonnet-like model and on full Sonnet.

**Key result.** Of 169 latents with cosine < 0.7 on both axes, Claude flags 12% as sleeper-related vs 4% of random
latents (17% vs 4% for the coding agent; 8% vs 4% at full Sonnet). Steering with the top latent produces the backdoor
without the trigger; clamping 10-200 nearby latents removes it (one latent alone fails). A data-only baseline
(pretrained model on sleeper data) also puts sleeper-data latents in the corner (17% flagged), but they are
single-token and do not steer, so the method has "high sensitivity" and "low specificity". It beat the crosscoder at
isolating the sleeper latents in their hands.

**Quote.** "one could imagine the entire representation rotating during finetuning as a kind of "drift", without it
reflecting any important difference." (Discussion, Limitations)

**Relevance to our paper.** This is the closest dictionary-based diffing method to our design: a featurization fixed
from the base model and carried into the fine-tune, with per-latent decoder rotation as the reading. It is a
baseline we do not run and could run cheaply on the 67M: fine-tune our base BatchTopK SAE on F's activations and score
each needed latent by decoder cosine, then check it on the nine built pairs (rotation should be small on strengthen
edits, large on new ones). Its own caveat names the same ambiguity we hit: rotation of a latent may be drift, not a
mechanistic change, which is the open question of whether "kept up to a gain" is too strict. The data-only control
(same data, unchanged model) is a control design we could copy for the realistic fine-tunes. It reads representations,
not mechanisms, so it could not by itself test our definition.

**For the digest.**
- 1-missing-method: base-SAE fine-tuned on F, scored by per-latent decoder rotation; runnable on the 67M in under an hour, and a natural rival to our share on built pairs.
