---
slug: oh2026-sharpening-tax
title: "Sharpening Tax in Post-Training"
authors: Oh et al.
year: 2026
id: 2610.01509
tags: [EH, SUPPORTS, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF arxiv_2610.01509.pdf (v1, Oct 2026) via pdf-text: abstract, §1-4 (pp. 1-7: agentic pass@k, bimodalisation, Sharpening Tax definition and results), §6 opening; PTGS (§5) and appendices only skimmed"
digest_buckets: [4-supports]
importance: 3
---

# Sharpening Tax in Post-Training (Oh 2026)

**What it does.** Tests the sharpening hypothesis on agentic multi-turn tool-use benchmarks (BFCL v4 multi-turn,
WebShop, ACEBench), where base models might be expected to lack the skills. It uses 14 public base/post-trained
checkpoint pairs from Gemma-4, Ministral-3, Qwen2.5 and Qwen3.5 (3B to 35B). The base models run in a light prompting
harness, and pass@k is measured up to k = 128. It defines "Sharpening Tax", the drop in area under the pass@k curve
(raw or normalised) from base to post-trained model, and proposes a per-prompt temperature sampler (PTGS) for RL.

**Key result.** Post-trained models win at pass@1, but harness-equipped base models scale faster in k and usually cross
over: for example over 85% pass@128 on WebShop for the gemma-4-31B base against 56% for the post-trained model. The
crossover comes earlier for larger models. Post-training "bimodalises" per-task success. On WebShop with gemma-4-31B,
the share of tasks solved only sometimes falls from 87.6% to 30.0%, the share always solved goes from 0% to 26%, and
the share never solved goes from 12.4% to 44%. TaxS(128) > 0 in 36 of 42 model-benchmark cases.

**Quote.** "post-training pushes tasks toward two extremes, always solved or never solved" (Abstract)

**Relevance to our paper.** It extends the behavioural EH evidence (Yue et al.) from maths and code to agentic tasks,
on general post-training pipelines rather than one RL run. That is useful for the related-work sentence saying the
evidence so far is behavioural. It does not look inside the model, gives no known answer, and its metric is a pass@k
summary, not a reading of the update. Little direct use beyond one citation.

**For the digest.**
- 4-supports: behavioural support that general post-training mostly sharpens base behaviour, even on agentic tool use, across 14 public base/instruct pairs.
