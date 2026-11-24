---
slug: kassem2025-mneme
title: "Reviving Your Mneme: Predicting the Side Effects of LLM Unlearning and Fine-Tuning via Sparse Model Diffing"
authors: Kassem et al.
year: 2025
id: 2507.21084
tags: [DIFFING, BASELINE, ALIGNMENT]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: "arXiv HTML: abstract, sec. 1-3 (method), sec. 4-6 (WMDP, EM, benign fine-tuning results)"
digest_buckets: []
importance: 3
---

# Reviving Your Mneme: Predicting the Side Effects of LLM Unlearning and Fine-Tuning via Sparse Model Diffing (Kassem 2025)

**What it does.** MNEME trains a BatchTopK crosscoder (expansion 32, layer 14, about 200M tokens of Pile or LMSYS-Chat)
on a base and an edited model. It labels each latent amplified, suppressed or unchanged by latent scaling (or decoder
norms), auto-interprets latents with Llama-3.1-70B, and maps the descriptions to topic categories. The categories are
scored against benchmark categories that degraded. Edits: RMU unlearning on WMDP (Llama-3.2-3B, Llama-3-8B, Zephyr-7B),
emergent misalignment from insecure code (Qwen2.5-Coder-7B), and Qi et al.'s harmful/benign fine-tunes
(Llama-2-7B-Chat).

**Key result.** Predicted affected MMLU categories match the degraded ones with 92-98% accuracy on WMDP unlearning, 92.2%
on MMLU-Pro and 68.2% on the EM harm categories, against about 50% for random and low recall for a GPT-4o guess. In
unlearning, the decoder-norm distribution moves toward the base side as forgetting grows.

**Quote.** "MNEME achieves up to 95% accuracy in predicting side effects" (abstract)

**Relevance to our paper.** An applied crosscoder-diffing pipeline, scored against behavioural ground truth (which topics
got worse), not against mechanisms whose reuse is known. It is the same family as our crosscoder baseline, so it adds no
new method; it is one more instance of the gap we name, a diffing method validated only on outputs. Its intro restates
Prakash et al. as "fine-tuning typically alters existing capabilities rather than adding new ones", the received view we
test. Little else bears on us.

**For the digest.** (none)
