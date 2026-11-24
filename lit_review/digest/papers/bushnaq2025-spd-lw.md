---
slug: bushnaq2025-spd-lw
title: "Stochastic Parameter Decomposition (paper and LessWrong linkpost)"
authors: Bushnaq et al. (Braun, Sharkey)
year: 2025
id: 2506.20790
tags: [DECOMP, BACKGROUND]
primary_tag: DECOMP
in_our_bib: bushnaq2025spd
cited_in_draft: yes
themes: [decompositions]
read_from: "arXiv HTML: abstract, intro, method (rank-one subcomponents, causal-importance function, stochastic masking), toy-model list, limitations; LessWrong post (2025-06-27) read via WebFetch: abstract and links plus a comment thread"
digest_buckets: []
importance: 3
---

# Stochastic Parameter Decomposition (Bushnaq 2025)

**What it does.** It replaces APD's gradient attributions and top-k with a learned causal-importance function. Each
weight matrix is decomposed into rank-one subcomponents, and a small network predicts, per input, how ablatable each one
is (g_i in [0, 1]). Training samples random masks between g_i and 1, so that every allowed combination of ablations must
leave the output unchanged, and it penalises g toward zero. Tested on the APD toy models (superposition, compressed
computation, cross-layer distributed representations) and two harder ones.

**Key result.** SPD recovers the ground-truth mechanisms in all the toy models, is less sensitive to hyperparameters
than APD, and avoids the shrinkage of learned parameters. It needs an extra clustering step to group rank-one
subcomponents into full components, and it was not yet shown on large models. The LessWrong post is the announcement
and adds nothing on fine-tuning or diffing.

**Quote.** "we define the causal importance of a subcomponent as how ablatable it is on a given datapoint." (Sec. 2)

**Relevance to our paper.** Background for VPD, which the draft uses; its causal-importance function is the ancestor of
the Γ that our re-gating share reads. It is cited correctly. Cite the paper, not the post. It has nothing on
fine-tuning, so the draft's claim that a decomposition "has not been used to diff a fine-tune before" is not affected by
this paper.

**For the digest.**
