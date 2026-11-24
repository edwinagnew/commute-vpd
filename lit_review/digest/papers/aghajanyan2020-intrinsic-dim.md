---
slug: aghajanyan2020-intrinsic-dim
title: "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"
authors: Aghajanyan et al.
year: 2020
id: 2012.13255
tags: [EH, DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, elicitation]
read_from: "arXiv HTML full (abstract, sec. 3 method, sec. 4-5 results)"
digest_buckets: [4-supports]
importance: 4
---

# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (Aghajanyan 2020)

**What it does.** Measures the intrinsic dimension d90 of fine-tuning pretrained models (BERT, RoBERTa, BART, T5, XLNet
and others) on GLUE-style tasks: the smallest number of parameters, randomly projected into the full weight space with a
Fastfood transform (with a per-layer scale in the "SAID" variant), that reaches 90% of full fine-tuning. It re-pretrains
RoBERTa-base from scratch and measures d90 at checkpoints, and compares d90 across model sizes.

**Key result.** RoBERTa-Large reaches 90% of full fine-tuning on MRPC with about 200 parameters and on QQP with about
800. d90 falls monotonically over pre-training, for all six tasks, without the model seeing them. Larger models have
lower d90, and lower d90 goes with better held-out accuracy; the paper derives generalisation bounds from d90.

**Quote.** "pre-training implicitly minimizes intrinsic dimension" (abstract)

**Relevance to our paper.** The standard citation for "fine-tuning updates are low-dimensional", and a behavioural
(optimisation) argument for the elicitation hypothesis: pre-training builds a framework in which a task needs only a
few hundred directions. It cannot tell reuse from new: a random low-dimensional subspace could still build a new
computation, and the paper never looks at mechanisms. Worth one sentence in related work beside our update effective
rank and LoRA/IA3 refits; it is the precedent for measuring update size or dimension as a proxy for "how much was
learned", which our results say tells an update from noise but not reuse from new.

**For the digest.**
- 4-supports: behavioural, weight-space support for the elicitation framing (pre-training lowers the dimension a task needs); no mechanism-level reading.
