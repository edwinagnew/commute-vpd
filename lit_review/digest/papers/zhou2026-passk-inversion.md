---
slug: zhou2026-passk-inversion
title: "When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion"
authors: Todd Y. Zhou
year: 2026
id: 2607.20543
tags: [EH, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF (arXiv v1, Jul 2026) pages 1-12: abstract, regimes (Sec. 4), two-mode theory (Sec. 5), PBA method (Sec. 6), experiments and controls (Sec. 7, Tables 2-6)"
digest_buckets: []
importance: 2
---

# When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion (Zhou 2026)

**What it does.** Explains why RLVR can raise pass@1 while lowering pass@256. Prompts are split by frozen-base success
rate into solved-easy, reachable, boundary (rare but recoverable correct answers) and out-of-reach. A two-mode model
argues that on boundary prompts finite GRPO groups rarely see a correct rollout, so shared-parameter sharpening commits
to the wrong mode. Per-Problem Base Anchoring (PBA) applies GRPO only to prompts with a base success in 8 samples and a
KL to the base elsewhere. Qwen2.5-7B, GRPO on Omni-MATH, three seeds.

**Key result.** GRPO: pass@1 10.2 -> 25.1 but pass@256 69.1 -> 68.3, with boundary prompts losing 39 points at pass@256
(654 boundary prompts lost). PBA: pass@1 29.0 and pass@256 73.0, boundary lost prompts down 7.2x to 91. Global KL,
entropy bonus, random masking and dropping hard prompts recover less.

**Quote.** "RLVR without external exploration primarily reweights trajectories sampled from the student distribution."
(Sec. 2)

**Relevance to our paper.** A behavioural and theoretical account of RL as re-weighting of the base distribution, with
coverage loss localised to rare-mode prompts. It states the re-weighting view as an assumption rather than testing it
internally, and has no method or known-answer case usable by us. Of little use beyond a possible cite in an
elicitation-versus-sharpening sentence.

**For the digest.**
