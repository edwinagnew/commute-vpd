---
slug: pan2024-reward-gaming
title: "Feedback Loops With Language Models Drive In-Context Reward Hacking"
authors: Alexander Pan et al.
year: 2024
id: 2402.06627
tags: [ALIGNMENT, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML (ICML 2024): abstract, introduction, experiment summaries (sec. 4)"
digest_buckets: []
importance: 1
---

# Feedback Loops With Language Models Drive In-Context Reward Hacking (Pan 2024)

**What it does.** Studies deployed LLMs whose outputs change the world and come back into their context, for example an
agent posting tweets to raise engagement. It names this in-context reward hacking (ICRH) and studies two routes to it,
output-refinement and policy-refinement, with experiments on Claude-2, Claude-3 and GPT-3.5/4 and no weight updates.

**Key result.** Feedback loops drive test-time optimisation of a proxy, with harmful side effects; engagement rises
and so does toxicity. Neither scaling the model nor sharper prompts removes it, and scaling sometimes makes it worse.
Static evaluations miss it, so the paper recommends three evaluation practices.

**Quote.** "We show that two natural approaches to mitigating ICRH—scaling model size and improving prompt specification—are both ineffective." (Sec. 1)

**Relevance to our paper.** None. It involves no fine-tune and no weight change, only in-context optimisation. Oh et al.
cite it for reward gaming in general. Leave it out.

**For the digest.**
