---
slug: ghilardi2024-sae-layerwise-transfer
title: Accelerating Sparse Autoencoder Training via Layer-Wise Transfer Learning in Large Language Models
authors: Ghilardi et al.
year: 2024
id: https://doi.org/10.18653/v1/2024.blackboxnlp-1.32
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: ACL Anthology PDF pages 1-8 (abstract, background, method, results 4.1-4.3, conclusions, limitations) via fleet/pdf-text
digest_buckets: []
importance: 1
---

# Accelerating Sparse Autoencoder Training via Layer-Wise Transfer Learning in Large Language Models (Ghilardi 2024)

**What it does.** Asks whether an SAE trained on one layer's residual activations can initialize the SAE for an
adjacent layer of the same model, to save training compute. On Pythia-160M (12 layers), it measures zero-shot
cross-layer reconstruction (CE-loss score of SAE_j on layer i) and then fine-tunes forward-transferred (from layer i−1)
and backward-transferred (from layer i+1) SAEs, comparing reconstruction, faithfulness/completeness, max mean cosine
similarity to scratch SAEs, and human interpretability scores across checkpoints. Only one model; no fine-tuned models.

**Key result.** Zero-shot transfer between adjacent layers is partial (better onto the preceding layer). Transferred SAEs
fine-tuned with about a tenth of the tokens (100M of 1B) match or beat scratch SAEs on reconstruction, backward transfer
being best; many features keep the identity of the initializing SAE's features ("Feature Transfer").

**Quote.** "although transferability between layers exists, it remains partial and, potentially, not completely reliable
for downstream applications." (Sec. 4.1)

**Relevance to our paper.** Essentially none: it is SAE training efficiency across layers of one model, not a base vs
fine-tune comparison and not about mechanisms. It mentions Kissane et al.'s base-to-chat transfer only as the
"inter-model" case it does not study. Skip.

**For the digest.** (none)
