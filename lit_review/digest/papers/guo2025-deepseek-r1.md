---
slug: guo2025-deepseek-r1
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: Daya Guo et al. (DeepSeek-AI)
year: 2025
id: 2501.12948
tags: [EH, BACKGROUND]
primary_tag: EH
in_our_bib: deepseek2025r1
cited_in_draft: yes
themes: [elicitation]
read_from: "arXiv HTML v2 (Jan 2026): abstract, R1-Zero training and results incl. the 'aha moment', Appendix F.1 (distillation vs RL), Appendix G.1 key findings"
digest_buckets: [4-supports]
importance: 4
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (Guo 2025)

**What it does.** Trains DeepSeek-V3-Base with large-scale GRPO on verifiable rewards and no SFT (R1-Zero), then with a
multi-stage cold-start pipeline (R1). It distils R1's traces into smaller Qwen and Llama models, and compares that with
running the same large-scale RL directly on Qwen2.5-32B-Base.

**Key result.** R1-Zero's AIME 2024 pass@1 rises from 15.6% to 77.9% over RL (86.7% with self-consistency). The paper
says RL "facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification".
Two findings fit elicitation as well. First, distilling R1 beats RL on a small base (R1-Distill-Qwen-32B 72.6 against
Qwen2.5-32B-Zero 47.0 AIME pass@1). Second, RL on 7B dense and 16B MoE bases "consistently failed to yield meaningful
improvements", and gains appeared only with larger, "more capable" bases (G.1, "The importance of base checkpoint").

**Quote.** "Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories." (Abstract)

**Relevance to our paper.** The draft cites it once in main.tex as the view that "recent rapid improvements in LLM
capabilities are attributed to scaling reinforcement learning", which is a fair use. The paper's own word,
"incentivized", and its finding that RL works only when the base checkpoint is strong enough both lean toward
elicitation. The intro could say so: it sharpens the tension the intro sets up, since the paper usually cited for "RL
creates capability" reports that the capability depends on the base. It gives no mechanistic evidence, no known answer
and no method we could run.

**For the digest.**
- 4-supports: its own key findings (RL fails on weak bases; distillation beats RL on small ones) are behavioural evidence that the RL gains depend on the base; the draft cites it only as the "RL creates" side.
