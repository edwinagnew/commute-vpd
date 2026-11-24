---
slug: wen2026-rlvr-correct-reasoning
title: "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs"
authors: Wen et al.
year: 2026
id: 2506.14245
tags: [EH, CONTRADICTS, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF arxiv_2506.14245.pdf (v2, Oct 2025) via pdf-text: abstract, §1-2, §3.1 (Pass@K vs CoT-Pass@K on DAPO-Qwen-32B), §4 theorem statement and assumptions; training-dynamics and SFT-on-CoT sections only skimmed; appendices not read"
digest_buckets: [4-contradicts]
importance: 4
---

# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs (Wen 2026)

**What it does.** Re-examines the pass@k evidence that RLVR only improves sampling efficiency (Yue et al. 2025). It
adds CoT-Pass@K, which counts a sample only if both the answer and the chain of thought are correct, judged by
DeepSeek-R1-0528-Qwen3-8B with several verifications and manual checks. It compares Qwen2.5-32B with DAPO-Qwen-32B on
maths, and R1-Distill-Qwen-7B with AceReason-Nemotron-7B on LiveCodeBench. It proves that, under a "logic prior"
(correct chains of thought are likelier to give correct answers), GRPO's expected advantage is positive for correct
chains and negative for incorrect ones. It also tracks this during a DAPO reproduction.

**Key result.** By plain Pass@K the base model catches up with the RLVR model, as Yue et al. found. By CoT-Pass@K, on
AIME 2024 and 2025 the RLVR model stays ahead at every K up to 1024, most clearly on AIME 2025, which is after the
base model's cutoff. On MATH-500 and AMC23 the effect is small. Code shows extended pass@k too. Correct reasoning is
rewarded from early in training.

**Quote.** "the CoT-Pass@K results on AIME 2024 and AIME 2025 reveal a consistent and significant performance gap
between the models across all values of K (up to 1024)." (§3.1)

**Relevance to our paper.** Our draft cites Yue et al. for the behavioural selection view. This paper shows that
evidence is confounded: base models reach right answers by wrong reasoning, which inflates pass@k. That supports our
gap statement, that behavioural tests of elicitation are fragile, and the draft should cite it beside Yue as the
direct rebuttal. Its theorem is itself a selection argument: RLVR raises the probability of correct reasoning the base
model already holds, given a prior. So its "extends the boundary" claim is compatible with re-weighting at the level of
reasoning paths. That is worth noting when we say what behaviour can and cannot tell. No internal analysis, no
known answer.

**For the digest.**
- 4-contradicts: with chain-of-thought correctness checked, RLVR extends the base model's reasoning boundary on AIME, against the pass@k reading of Yue et al. that the draft relies on.
