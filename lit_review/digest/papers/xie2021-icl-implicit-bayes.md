---
slug: xie2021-icl-implicit-bayes
title: An Explanation of In-context Learning as Implicit Bayesian Inference
authors: Sang Michael Xie et al.
year: 2021
id: 2111.02080
tags: [EH, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "PDF pages 1-9 (abstract, intro, framework, GINC simulations)"
digest_buckets: []
importance: 2
---

# An Explanation of In-context Learning as Implicit Bayesian Inference (Xie 2021)

**What it does.** A theory paper with a synthetic dataset (ICLR 2022). It models pretraining documents as a mixture of HMMs,
each driven by a latent "concept", and proves that in-context learning arises when the model infers the concept shared by
the examples in a prompt, even though prompts differ from pretraining text. GINC, a small synthetic dataset, lets
Transformers and LSTMs trained on it show in-context learning.

**Key result.** In the theory, in-context learning is implicit Bayesian inference over latent concepts learned in
pretraining. On GINC it emerges from the long-range coherence of the pretraining data. The simulations also reproduce
three effects seen in large models: in-context accuracy improves with scale at equal pretraining loss, it is sensitive to
example order, and zero-shot sometimes beats few-shot.

**Quote.** "During pretraining, the language model (LM) implicitly learns to infer a latent concept ... shared across sentences in a document." (Figure 1 caption)

**Relevance to our paper.** It is background for the selection view: the prompt picks out a capability learned in
pretraining, which is the in-context analogue of EH and the intuition behind URIAL. It says nothing about weight updates,
fine-tuning or mechanisms. Cite it only if the related-work section discusses prompting as elicitation.

**For the digest.**
