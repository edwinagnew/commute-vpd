---
slug: su2026-agentic
title: Scaling Agents via Continual Pre-training
authors: Liangcai Su et al.
year: 2026
id: 2509.13310
tags: [EH, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: arXiv HTML: abstract, introduction, ablation and SFT-loss sections (3.3-3.4 and loss analysis); identified from Oh et al. 2026's bibliography ("Su et al. (2026) ... Scaling agents via continual pre-training", ICLR 2026)
digest_buckets: []
importance: 2
---

# Scaling Agents via Continual Pre-training (Su 2026)

**What it does.** Argues that post-training a general base model for deep-research agent tasks forces it to learn
agentic capabilities and alignment at once, and inserts "Agentic Continual Pre-training" (synthesised planning,
reasoning and multi-step decision data, up to 315B tokens) between pre-training and post-training of
Qwen3-30B-A3B-Base, giving AgentFounder-30B.

**Key result.** AgentFounder-30B reaches 39.9% on BrowseComp-en, 43.3% on BrowseComp-zh and 31.5% Pass@1 on HLE.
The same SFT data on the agentic-CPT base beats the plain base by 5.75-6.45% on average, and SFT loss is lower (final
0.7953 vs 0.8656), falling as CPT data grows.

**Quote.** "general-purpose foundation models lack agentic inductive biases, forcing post-training to simultaneously learn capabilities and alignment, creating inherent optimization conflicts." (Section 1)

**Relevance to our paper.** Little. It is a training-recipe paper whose framing (capabilities should come from
pre-training, post-training should only align them) restates the elicitation hypothesis as engineering advice. Oh et
al. cite it for agentic skills being believed to be learned in post-training. No internals.

**For the digest.**
