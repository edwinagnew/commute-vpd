---
slug: boughorbel2025-model-diffing-leaderboard
title: "Beyond the Leaderboard: Understanding Performance Disparities in Large Language Models via Model Diffing"
authors: Boughorbel et al.
year: 2025
id: 2509.18792
tags: [DIFFING, BASELINE, EH, SUPPORTS]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, elicitation]
read_from: "arXiv HTML v1: abstract, sections 1-4 (methodology, design iteration, findings, discussion); appendices not read"
digest_buckets: [4-supports]
importance: 2
---

# Beyond the Leaderboard: Understanding Performance Disparities in LLMs via Model Diffing (Boughorbel 2025)

**What it does.** Applies BatchTopK crosscoders with Minder et al.'s Latent Scaling (layer 20, 114,688 latents, k =
100, 200M tokens of FineWeb plus LMSys) to Gemma-2-9b-pt, Gemma-2-9b-it and Gemma-2-9b-it-SimPO. Latents unique to
one model are labelled by Claude 3 Opus from top-activating documents into a 30-category taxonomy, and category
frequencies are compared between the it and SimPO models.

**Key result.** A crosscoder on it versus it-SimPO finds no model-unique latents (relative norms mostly 0.3 to 0.6).
Against the pretrained base there are 113 latents unique to it and 92 unique to SimPO. By category counts SimPO shows
more instruction-following (+151.7%), multilingual (+43.8%) and safety (+32.8%) latents, and fewer
hallucination-management (-68.5%) and self-reference (-44.1%) latents.

**Quote.** "They may modify surface-level generation preferences or alignment signals rather than introducing
fundamentally new internal representations." (Section 2.3)

**Relevance to our paper.** It is an applied use of the crosscoder baseline we already adapt from Minder et al., with
no known answer and no causal check: the percentages are changes in counts of LLM-labelled latents, from about 100
latents per model. The one finding that bears on us is the null between it and SimPO: a preference-optimisation step
on an instruction-tuned model leaves no crosscoder-exclusive latents. The authors read that as re-weighting, but by our
own results a crosscoder cannot tell reuse from new at fine-tune size, so the null is just as consistent with the tool
being blind. At most a citation as an example of crosscoder diffing used to explain benchmark gaps.

**For the digest.**
- 4-supports: weak support for "preference tuning adds no new representations" (it vs SimPO null), but read with a tool we show is blind at that size.
