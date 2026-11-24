---
slug: wu2025-invisible-leash
title: "The Invisible Leash? Why RLVR May or May Not Escape Its Origin"
authors: Fang Wu et al.
year: 2025
id: 2507.14843
tags: [EH, SUPPORTS, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF (arXiv v5, Oct 2026) pages 1-8: abstract, framework (Sec. 2), results (Sec. 3, Tables 1-4), start of entropy section"
digest_buckets: [4-supports, 5-cite-fix]
importance: 4
---

# The Invisible Leash? Why RLVR May or May Not Escape Its Origin (Wu 2025)

**What it does.** Asks whether RLVR expands what a model can solve or mostly re-weights solutions the base model already
reaches. It defines "empirical support" (questions with at least one verified success in n samples) and counts, question
by question, preservation, expansion and shrinkage between base and RLVR checkpoints. Models: ProRL-1.5B (from
DeepSeek-R1-Distill-Qwen-1.5B), Nemotron/AceReason 7B-14B, Skywork-OR1-7B, Phi4-Reason-Plus-14B, OLMo-2-1B, a VLM; maths,
logic, factual QA and code; budgets up to k = 8192. Also an SFT vs DAPO comparison on Qwen2.5-Math-7B and a DAPO
checkpoint trace.

**Key result.** RLVR keeps most base successes (SRR 0.93-0.99) but newly solved questions are at most ~2% of its successes
(NDR <= 0.02 for the 1.5B-14B models), and losses exceed gains in every model-level aggregate (ProRL-1.5B-v2 loses 175
questions, gains 48). With the same base and data, SFT keeps or grows coverage on all six benchmarks while DAPO shrinks
it on four. Some RLVR models raise token entropy while narrowing final answers.

**Quote.** "The current RLVR recipe primarily preserves rather than expands the base model's solution coverage." (Sec. 1)

**Relevance to our paper.** Behavioural evidence for the elicitation hypothesis on RL, at many scales: RL mostly sharpens
the base distribution, with a small, real expansion on some tasks (arc_1d, boxnet). It is exactly the kind of
"behavioural" evidence our introduction says cannot settle EH, and its SFT-vs-RL contrast (SFT expands more) is a useful
foil to our RL-on-repeat reading (RL reads higher ρ, 0.05-0.22) and to the Qwen RL vs SFT appendix. It looks at no
internals, so it is no baseline and no known-answer case. Our related-work section asks for an
elicitation-versus-sharpening literature; this paper (with Yue et al. 2025) is a natural cite there.

**For the digest.**
- 4-supports: large-scale behavioural evidence that RLVR re-weights base solutions (NDR <= 0.02), cite beside Yue 2025 in "post-training as selection".
