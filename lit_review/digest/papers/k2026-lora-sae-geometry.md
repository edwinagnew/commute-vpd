---
slug: k2026-lora-sae-geometry
title: "Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis of Representational Divergence in Fine-Tuned Language Models"
authors: Prasanth K K
year: 2026
id: 2605.28896
tags: [DIFFING, FT-INTERP]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes]
read_from: "arXiv HTML: full text (abstract, method, setup, results sec. 5, ablations, discussion and limitations)"
digest_buckets: []
importance: 2
---

# Feature Geometry of LoRA Adapters (Prasanth K K 2026)

**What it does.** It trains LoRA adapters (ranks 4 to 32, on q/k/v/o, Alpaca 10k, 3 epochs) on Gemma-2-9B. It takes
the residual-stream difference h_adapted − h_base at six layers and trains a fresh "delta SAE" on those differences. It
then compares the delta SAE's decoder directions with Gemma Scope's base-model SAE by max cosine, principal angles
between top-256 subspaces, and linear CKA.

**Key result.** Delta-SAE directions have mean max cosine about 0.071 to any Gemma Scope feature. Principal angles
are about 74°. Gemma Scope reconstructs the deltas with relative error above 1.0 everywhere, while the delta SAEs reach
0.34 to 0.65. The author reads this as LoRA working in a feature subspace separate from the base model's. The author
also reports that the adapters' outputs equal the base model's on test prompts, and that there is no base-vs-base SAE
control.

**Quote.** "the adapters produce outputs identical to the base model on test prompts" (Sec. 7.4)

**Relevance to our paper.** On its face it says LoRA writes new directions, which sounds like our "turn rather than
scale" finding. It does not support it. The adapters change no behaviour, so there is no fine-tuned behaviour whose
mechanism could be reused or new. A base SAE trained on activations, bias included, is the wrong tool for difference
vectors, so error above 1.0 is expected. And 0.07 max cosine has no seed-to-seed reference. It is not a known-answer
case and not a baseline worth running. At most it is an example of the activation-delta diffing we argue cannot tell
reuse from new.

**For the digest.** (no bucket)
