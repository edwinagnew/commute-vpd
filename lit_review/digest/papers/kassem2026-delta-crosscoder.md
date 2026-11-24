---
slug: kassem2026-delta-crosscoder
title: "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes"
authors: Kassem et al.
year: 2026
id: 2603.04426
tags: [BASELINE, DIFFING, CITE-FIX, EH, ALIGNMENT]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, elicitation]
read_from: arXiv HTML abstract, sections 1, 3-8 (method, setup, all results, discussion); appendices not read
digest_buckets: [1-missing-method, 5-cite-fix]
importance: 6
---

# Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes (Kassem 2026)

**What it does.** Changes the BatchTopK crosscoder so that it finds the small, sparse shifts that narrow fine-tunes
make. It reserves 80% of latents as non-shared, with a smaller top-K budget for them. An extra loss reconstructs the
activation difference Δ = b − a through the non-shared latents only. Training pairs are general prompts answered by
both models, each answer run through both models. It is tested on 10 model organisms (false-fact document fine-tuning,
taboo word, subliminal learning, emergent misalignment) on Gemma, Llama and Qwen, 1B to 9B. The top-3 non-shared latents
by relative decoder norm are validated by steering and max-activation examples.

**Key result.** It finds a causally effective latent in 10 of 10 organisms, against 6 of 10 for Designated Shared
Features crosscoders and 4 of 10 for BatchTopK crosscoders (sizes 200 and 400). Graded by an LLM, it matches the
activation difference lens on emergent misalignment and false facts, ties on taboo, and is below it on subliminal
learning. On two identical models, the relative-norm tail reaches only 0.506, with no causal latents. Steering the base
model with the recovered latent induces the fine-tuned behaviour (misalignment, false beliefs).

**Quote.** "Crosscoders ... consistently fail to recover causally relevant features in this regime. This failure is
structural: joint reconstruction prioritizes high-frequency shared features while suppressing sparse, low-magnitude
shifts" (Sec. 1)

**Relevance to our paper.** This is the strongest crosscoder baseline for our setting, and the draft does not cite it.
Our crosscoder baseline is the BatchTopK crosscoder of Minder et al., and it misses a planted control up to about 100
times fine-tune size. This paper says that failure is structural and gives a fix. Under the project's rule of running a
baseline with the best published recipe, a reviewer will ask why Delta-Crosscoder was not run. It runs on our setting:
it needs only residual activations of M and F on paired inputs, cheaply on the 67M, and a reuse share could be read
from its non-shared latents as we do for the BatchTopK one. Its "latent capability present in the base model" claim
rests on steering the base model with an F-derived direction. That is not evidence of reuse in our sense, because the
added vector can carry the learned content itself. So it is not a known-answer case.

**For the digest.**
- 1-missing-method: Delta-Crosscoder (delta loss, reserved non-shared latents, paired generations) is built for the narrow-fine-tune regime where our BatchTopK crosscoder fails; runnable on the 67M; cite and ideally run.
