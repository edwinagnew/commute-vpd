---
slug: bushnaq2026-vpd-linkpost
title: "[Linkpost] Interpreting Language Model Parameters (adVersarial Parameter Decomposition, VPD)"
authors: Bushnaq et al. (Braun, Clive-Griffin, Bussmann, Hu, Ivanitskiy, Linsefors, Sharkey)
year: 2026
id: 2610.20748 (paper); https://www.lesswrong.com/posts/eAQZaiC3PcBhS4HjM
tags: [DECOMP, CITE-FIX, BACKGROUND, SUPPORTS]
primary_tag: DECOMP
in_our_bib: bushnaq2026vpd
cited_in_draft: yes
themes: [decompositions, interpreting-finetunes]
read_from: "LessWrong linkpost full (via LW GraphQL); arXiv HTML of 2610.20748: abstract, Sec. 3.1 target model and decomposition stats, Sec. 7.5 limitations and future work, Sec. 7.6.1 related work"
digest_buckets: [4-supports, 5-cite-fix]
importance: 7
---

# Interpreting Language Model Parameters: adVersarial Parameter Decomposition (Bushnaq 2026)

**What it does.** Introduces VPD, which decomposes a language model's weight matrices into rank-one parameter
subcomponents that sum to the weights, with a learned causal-importance function saying which subcomponents each token
needs, and trains them so the output survives ablation of the unneeded ones, including adversarially chosen ablations
(PGD). Applied to Goodfire's four-layer, 67M-parameter Pile model (embeddings excluded), and used to build attribution
graphs with subcomponents as nodes; compared against per-layer and cross-layer transcoders.

**Key result.** The 24 weight matrices are decomposed into 38,912 rank-one subcomponents, of which ~10,000 are alive;
each token uses on average 205 subcomponents (2.1% of the alive ones). The post says attention layers are decomposed,
VPD shows no feature splitting, it compares "somewhat favourably" with transcoders, and the adversarial ablation was
"pretty important for faithfully identifying which nodes ... were causally important".

**Quote.** "This could allow us to express each gradient update to a model as a combination of upweighting,
downweighting, or modifying existing subcomponents, as well as creating new ones outside the span of the existing
subcomponent subspace." (paper, Sec. 7.5, "Interpreting parameter gradients to understand training and finetuning")

**Relevance to our paper.** This is our featurization's source and our 67M model's decomposition, so it is cited
already. The important point for the draft: Sec. 7.5 proposes, as future work, exactly our programme: project a
fine-tune's weight diff onto the subcomponents and read it as upweighting (rescaling, our "kept"), modifying (our
turned writes) or creating new directions (our outside share). The draft's claim that a decomposition "has not been used
to diff a fine-tune before" is still true as far as this paper goes, since it only proposes it. But the related-work
section should credit the idea to this passage and present our paper as the first test of it. Our finding that trained
fine-tunes turn writes rather than rescale them answers the authors' proposed split directly. The paper's caveat about
the adversary's robustness for aggregation is the opening for Angsten 2026 (separate entry), which bears on our
re-gating share.

**For the digest.**
- 4-supports: VPD's own authors propose reading a fine-tune diff as up/down-weighting vs modifying vs new subcomponents; our paper is the first test of that proposal (cite Sec. 7.5).
