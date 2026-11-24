---
slug: zhang2026-rl-activation-intensity
title: "Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs"
authors: Zhang et al.
year: 2026
id: https://proceedings.iclr.cc/paper_files/paper/2026/hash/67d1e25d35292dab646b58c0af77151a-Abstract-Conference.html
tags: [FT-INTERP, DIFFING, EH]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "ICLR 2026 proceedings PDF: abstract, intro, sec. 4 (settings, metrics, Table 1, results), Appendix B (temperature experiment)"
digest_buckets: []
importance: 3
---

# Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs (Zhang 2026)

**What it does.** It asks what online RL changes inside an LLM compared with its SFT starting point. It uses edge
attribution patching (EAP) between attention and MLP nodes on solutions that both models get right, and summarises
the edge-weight tensor by three statistics: mean absolute edge weight ("activation intensity"), the histogram entropy
of edge weights, and kurtosis. The pairs are about 7B: DeepSeek-Math instruct vs GRPO, Mistral MetaMath-SFT vs
Math-Shepherd PPO, R1-Distill-Qwen vs an RL successor, and Qwen2.5-7B SFT vs DPO. The data are GSM8K, MATH and College Math.

**Key result.** For the online-RL pairs, intensity and entropy mostly rise and kurtosis falls after RL (Table 1), with
exceptions at small truncation scales. The DPO pair does not follow the trend. The authors read this as RL engaging
more, and more diverse, pathways, and propose that RL "activates and reinforces latent or 'dormant' internal circuits".
The only causal check (Appendix B) is one temperature contrast (0.6 vs 1.0) in GRPO on Qwen2.5-3B-Instruct: the 0.6 run shows larger metric changes and reaches 51.2% on GSM8K. That is a correlation between two runs, not an intervention on the circuits.

**Quote.** "an overall increase in average activation intensity, indicating that more internal pathways are engaged and their signals become stronger" (Abstract)

**Relevance to our paper.** It is related work for RL post-training and the internals of real reasoning models, so it
is citable where the draft discusses RL (our RL repeat-task and Qwen maths readings). Its statistics are global
aggregates over attribution scores. They cannot say whether a given edge is a base mechanism kept or one built new,
and neither "stronger" nor "more pathways" is tested against any mechanism, so it is not a known-answer case. Its
"dormant circuits" story is an informal elicitation claim made from the sampling-distribution argument, not shown.
The pairs also differ in lineage and data beyond RL alone.

**For the digest.** (no bucket)
