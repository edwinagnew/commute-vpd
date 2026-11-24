---
slug: varadi2026-cross-seed-sae
title: "Cross-seed explainability using Procrustes conditioned Joint End-to-end Top-K Sparse Autoencoders"
authors: Váradi & Kmetty
year: 2026
id: 2607.08499
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, intro, method (sec. 3), experiments (sec. 5) text, limitations"
digest_buckets: []
importance: 1
---

# Cross-seed explainability using Procrustes conditioned Joint End-to-end Top-K Sparse Autoencoders (Váradi 2026)

**What it does.** It trains one joint Top-K SAE over the activations of two BERT models from different seeds. Seed B's
activations are first rotated onto seed A's by an orthogonal Procrustes fit on 500 examples. There are end-to-end KL
and downstream losses and a cross-seed code-matching loss. Features are counted as "universal" when they correlate at
r ≥ 0.7 across seeds. Five seed pairs; SST-2, Stanford Politeness, TweetEval Emotion.

**Key result.** The full pipeline gives the highest top-100 cross-seed correlation on all three datasets and the most
universal features on two of three, against post-hoc Hungarian-matched independent SAEs. Applying the rotation post
hoc to independent SAEs does not help consistently.

**Quote.** "applying the rotation post-hoc to independent SAEs yields no consistent benefit" (Sec. 5)

**Relevance to our paper.** Little. It compares seeds trained from scratch, not a base model and a fine-tune, on small
BERT classifiers. A joint SAE over two models is a crosscoder-like object, but we already engage crosscoders through
Minder et al.

**For the digest.** (no bucket)
