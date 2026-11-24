---
slug: jiralerspong2026-dedicated-feature-crosscoder
title: "Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs"
authors: Jiralerspong & Bricken
year: 2026
id: 2602.11729
tags: [BASELINE, DIFFING, SUPPORTS, DEFINITION]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, reuse-definitions]
read_from: arXiv HTML abstract, intro, sec 3.2.1 (toy model), sec 5 (discussion, limitations); appendix skimmed by contents
digest_buckets: [3-definition, 4-supports]
importance: 4
---

# Cross-Architecture Model Diffing with Crosscoders (Jiralerspong 2026)

**What it does.** Introduces Dedicated Feature Crosscoders (DFCs), which partition the dictionary into A-exclusive,
B-exclusive and shared index sets by construction (exclusive latents decode to only one model), with BatchTopK
sparsity. Tests on a toy model with 2048 planted concepts (2.5% exclusive per model) against a standard crosscoder and
the designated-shared-feature (DSF) crosscoder, then diffs models of different architectures: Llama-3.1-8B-Instruct vs
Qwen3-8B, and GPT-OSS-20B vs DeepSeek-R1-0528-Qwen3-8B, validating exclusive latents by steering.

**Key result.** On the toy model, DFCs recover more exclusive concepts than both baselines, most in the undercomplete
regime, at a higher false-positive rate; in all architectures exclusive concepts are learned later in training than
shared ones, which they read as the crosscoder's built-in prior against exclusive features. On real models they find
steerable exclusive latents such as "CCP alignment" in Qwen and "American exceptionalism" in Llama, the first robust
across seeds and the latter found in 4 of 5 runs. They report that base-vs-fine-tune diffs ran into "mirror features"
in preliminary runs.

**Quote.** "identifying a feature as "model-exclusive" refers to its specific representation, not necessarily the
model's underlying conceptual capability." (Sec 5.1)

**Relevance to our paper.** A clear statement, from crosscoder authors, that crosscoder exclusivity is a claim about
representations and not about capability or mechanism; this supports our point that no existing diffing test looks
inside the model, and is citable when we contrast our definition with crosscoder "exclusive" labels. The toy result
that exclusive concepts are learned late, against a prior favouring shared latents, is a mechanism for our finding
that the crosscoder share misses a planted control at fine-tune size. DFC is a stronger crosscoder variant we could
swap in for the Minder et al. BatchTopK baseline; the paper's own note that it struggles on base-vs-fine-tune pairs
suggests it would not rescue that baseline. Cross-architecture diffing is otherwise outside our setting.

**For the digest.**
- 3-definition: "model-exclusive" is scoped to a linear representation, not a capability; contrast with our mechanism-level "new".
- 4-supports: crosscoders carry a prior against exclusive features (learned late in toy runs), consistent with our crosscoder missing small planted changes.
