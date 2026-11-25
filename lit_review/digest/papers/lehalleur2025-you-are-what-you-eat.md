---
slug: lehalleur2025-you-are-what-you-eat
title: You Are What You Eat -- AI Alignment Requires Understanding How Data Shapes Structure and Generalisation
authors: Lehalleur et al. (Hoogland, Farrugia-Roberts, Wei, Gietelink Oldenziel, Wang, Carroll, Murfet)
year: 2025
id: 2502.05475
tags: [CITE-FIX, BACKGROUND, EH]
primary_tag: CITE-FIX
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: arXiv HTML full text (sections 1-8; position paper, no experiments)
digest_buckets: [5-cite-fix]
importance: 3
---

# You Are What You Eat: AI Alignment Requires Understanding How Data Shapes Structure and Generalisation (Lehalleur 2025)

**What it does.** A position paper from the singular-learning-theory group (Timaeus / Murfet). It argues that alignment
methods (SFT, RLHF, DPO, constitutional AI) only shape the data distribution, which shapes internal structure only
indirectly, and that safety assurance therefore needs a science of how data shapes internal structure and how
structure decides generalisation. Its support is a review: underspecification (D'Amour et al. 2022), deep-linear and
kernel results on the order in which data modes are learned, and SLT's internal model selection (a simpler,
higher-loss solution can dominate the posterior at finite n).

**Key result.** No experiments. The argument: networks with equal training performance can "compute their outputs in
essentially different ways and thus generalise differently", so evaluation cannot certify what was learned. Section
5.3 adds that safety fine-tuning is easily undone (Qi et al. 2023), suggesting post-training builds "shallow, easily
disrupted structures" and that "our understanding of how these techniques actually modify the internal
representations and algorithms of the models remains limited".

**Quote.** "It is our position that these differences in generalisation arise from differences in internal algorithms
and representations. Thus, if we care about what happens when the model is deployed in new contexts, then we must
study the structure that the model has learned and not merely its performance on the training set or evaluations"
(Section 2.3). Abstract: "standard testing and evaluation are insufficient for obtaining assurances of safety".

**Relevance to our paper.** A citable statement of our "Why mechanisms?" premise (main.tex, the paragraph starting "The
reasons for preferring a mechanistic definition to a behavioural one"): its first reason, that the same behaviour can
come from different mechanisms that generalise differently, is exactly this paper's Section 2.3, and it currently
rests on Shuttleworth et al. 2024 alone. It also fits the motivation (post-training monitoring: we do not know what
post-training changes inside the model, Section 5.3). It is a position paper, so it backs the framing, not a result;
it gives no definition of reuse, no method and no known-answer case. Its primary evidence for the point is D'Amour et
al. 2022 (underspecification, 2011.03395), which is not in our lit review or refs.bib and is the stronger empirical cite
beside it.

**For the digest.**
5-cite-fix: cite at "the same behaviour may be produced by different mechanisms ... that generalise differently" and at "behavioural evidence can be misleading" (main.tex "Why mechanisms?"; its abstract: evaluation is "insufficient"), with D'Amour 2022 as the empirical backing; its Section 5.3 (safety fine-tuning looks aligned but is undone easily, Qi et al. 2023) could fill the "[one more point?]" placeholder there.
