---
slug: kim2026-harness-post-training
title: The Interplay of Harness Design and Post-Training in LLM Agents
authors: Kyungmin Kim et al.
year: 2026
id: 2606.25447
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML: abstract, introduction, section 3 setup, section 4 analysis text (many numbers were in stripped math and are not quoted)
digest_buckets: []
importance: 1
---

# The Interplay of Harness Design and Post-Training in LLM Agents (Kim 2026)

**What it does.** Extends ALFWorld into a tool-calling benchmark with three harnesses of increasing informativeness
(tool descriptions, list of valid tools, carried items) and with tool-schema and task shifts, then post-trains
Qwen2.5-3B/7B-Instruct with GRPO and GiGPO under each harness.

**Key result.** Zero-shot success rises with harness informativeness, more for larger models. Post-training under an
informative harness beats adding that harness after post-training, and agents post-trained under a low-effort harness
collapse under a tool-schema change (generating non-existent tool names), in one case below the untrained model.

**Quote.** "a high-quality harness should be designed in advance and treated as an explicit part of the post-training recipe rather than as an inference-time add-on." (Section 1)

**Relevance to our paper.** Little. It concerns agent scaffolding and behavioural robustness; no model internals and
no bearing on reuse. It reached our list only as a citation in Oh et al. 2026.

**For the digest.**
