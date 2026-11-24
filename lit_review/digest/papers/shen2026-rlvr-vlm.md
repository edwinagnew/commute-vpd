---
slug: shen2026-rlvr-vlm
title: "Does RLVR Extend Reasoning Boundaries? Investigating Capability Expansion in Vision-Language Models (Ariadne)"
authors: Minghe Shen et al.
year: 2026
id: 2511.00710
tags: [EH, CONTRADICTS]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF (arXiv v4, Apr 2026) pages 1-6: abstract, intro, method (Sec. 3), main results (Sec. 4.3, Figs. 2-4)"
digest_buckets: [4-contradicts]
importance: 3
---

# Does RLVR Extend Reasoning Boundaries? Investigating Capability Expansion in VLMs (Shen 2026)

**What it does.** Tests the "RLVR only improves sampling efficiency" view in a low-support domain: synthetic maze
navigation from images (Ariadne, adapted from AlphaMaze), where difficulty is set by path length and number of turns.
Qwen2.5-VL-7B-Instruct (plus 3B and Qwen3-VL 4B/8B) is trained with GRPO and a dense prefix-matching reward on 4,700
short mazes (length <= 5, turns <= 2), with an SFT variant on the same data, and evaluated with pass@k up to 512 on
longer held-out mazes and zero-shot on MapBench and ReasonMap.

**Key result.** The base model's pass@k stays at 0% up to k = 512 for mazes of >= 5 steps or >= 2 turns; the RL model
solves some of them and grows with k on unseen lengths 6-7 and 3 turns, "tripling the effective reasoning horizon (from
2 to ~6 steps)". It also improves zero-shot on the two real-map benchmarks.

**Quote.** "RLVR extends the spatial reasoning boundary beyond the effective support of the base policy" (Sec. 2,
contributions)

**Relevance to our paper.** A behavioural counter-example to a strong elicitation hypothesis: where the base model has
"hard zero" support, RL with a shaped reward reaches new solutions. It argues the outcome depends on whether the domain
is well covered by pretraining, a useful nuance for our introduction. It reads no internals, and "zero at k = 512" does
not show a new mechanism (the base may have the pieces and lack only their composition), so it is not a known-answer
case for our definition. Little else for us.

**For the digest.**
- 4-contradicts: behavioural evidence that RL can solve tasks with zero base pass@512 (mazes), so behavioural EH is domain-dependent; mechanism not examined.
