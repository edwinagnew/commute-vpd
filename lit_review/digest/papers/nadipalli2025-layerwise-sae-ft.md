---
slug: nadipalli2025-layerwise-sae-ft
title: "Layer-Wise Evolution of Representations in Fine-Tuned Transformers: Insights from Sparse AutoEncoders"
authors: Nadipalli
year: 2025
id: 2502.16722
tags: [FT-INTERP, DIFFING, BACKGROUND]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing]
read_from: "arXiv PDF pages 1-5 (abstract, method, results 4.1-4.2); HTML not available"
digest_buckets: []
importance: 1
---

# Layer-Wise Evolution of Representations in Fine-Tuned Transformers: Insights from Sparse AutoEncoders (Nadipalli 2025)

**What it does.** Fine-tunes bert-base-uncased (and smaller BERTs) on three classification sets (IMDb sentiment,
Spotify review ratings, news topics), then compares base and fine-tuned models at layers 3, 6 and 12 by the cosine
similarity of dataset-averaged activations and by training separate small L1 SAEs (768 to 1024) on each model and
inspecting the token activations of the top-3 highest-variance latents.

**Key result.** Cosine similarity of averaged activations is near 1.0 in early layers and drops in late layers, most
steeply on the news task; smaller BERTs change less. Qualitative SAE readings: layer 3 keeps general features, layer 6
is mixed, layer 12 is task-specific.

**Quote.** "early layers primarily retain general representations, middle layers act as a transition between general and task-specific features, and later layers fully specialize in task adaptation." (Abstract)

**Relevance to our paper.** Little use. The comparison is of mean activations and of separately trained, unmatched SAEs
read by eye, with no controls or causal tests, so it is neither a baseline we should run nor evidence of reuse.

**For the digest.**
- (no bucket)
