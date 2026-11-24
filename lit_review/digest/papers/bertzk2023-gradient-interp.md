---
slug: bertzk2023-gradient-interp
title: "Training Process Transparency through Gradient Interpretability: Early experiments on toy language models"
authors: robertzk
year: 2023
id: https://www.lesswrong.com/posts/DtkA5jysFZGv7W4qP/training-process-transparency-through-gradient
tags: [DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics]
read_from: "LessWrong post via GraphQL htmlBody: intro, setup, examples of results, methodology (most of the post)"
digest_buckets: []
importance: 1
---

# Training Process Transparency through Gradient Interpretability (robertzk 2023)

**What it does.** A MATS 3.1 post. It trains a 12.2M-parameter, 3-layer transformer on WikiText2 with plain SGD, and
records every per-datum parameter update (about 882 GB). It then attributes outlier parameter shifts of individual
MLP neurons back to the training tokens that caused them, in the neuron basis.

**Key result.** Several late-layer neurons flagged this way (for example a "war" neuron and a sports neuron) flip the
predictions of those tokens when zero-ablated. Activations alone did not single them out. The method fails on early
layers, where ablation flips hundreds of unrelated tokens.

**Quote.** "we were not able to predict this behavior using activations alone." (Examples of results)

**Relevance to our paper.** Little. It is training from scratch, not fine-tuning, and has no base-vs-fine-tune
comparison. It does name the move we make, reading parameter shifts in another basis ("'virtual' parameter shifts
along the appropriate corresponding basis"), which it left for future work. That is a curiosity, not a citation we need.

**For the digest.** (no bucket)
