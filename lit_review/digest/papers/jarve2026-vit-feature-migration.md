---
slug: jarve2026-vit-feature-migration
title: Feature Evolution and Migration during Vision Transformer Training
authors: Järve et al.
year: 2026
id: 2608.20134
tags: [DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics]
read_from: arXiv HTML abstract, sections 1-3, 4.2-4.3 results, 5-6
digest_buckets: []
importance: 1
---

# Feature Evolution and Migration during Vision Transformer Training (Järve 2026)

**What it does.** Trains a ViT-Tiny on ImageNet-1k (300 epochs) and on a 10-superclass subset. It fits SAEs on CLS-token
activations at each epoch-layer pair and tracks features across time and depth by Spearman correlation of their
activation profiles. This is done both with a fixed SAE read across checkpoints and by matching independently trained
SAEs. It defines "migration" as a change in the layer where a feature is most detectable.

**Key result.** Migration happens mostly early in training. It runs mostly toward earlier layers, up to about −2.5
layers on average in the strongest regions. Migration toward deeper layers peaks at 4% of features. Deeper layers
stabilise earlier and more strongly than shallow ones.

**Quote.** "A low score may therefore indicate true disappearance of the feature, but it may also arise if the same
feature is present in a shifted, rotated, split, or otherwise reparameterized form." (Sec. 3.1)

**Relevance to our paper.** Little direct bearing: this is pre-training dynamics in a small vision model, not
fine-tuning, and it makes no reuse claim. Its quoted caveat is the same one that applies to our SAE featurization fixed
from M: a fixed dictionary read on F cannot tell a lost variable from a rotated one.

**For the digest.** (no bucket)
