---
slug: mukherjee2025-rl-subnetworks
title: Reinforcement Learning Finetunes Small Subnetworks in Large Language Models
authors: Mukherjee et al. (Yuan, Hakkani-Tür, Peng)
year: 2025
id: 2505.11711
tags: [FT-INTERP, DIFFING, EH, BASELINE, CITE-FIX]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "local PDF text (NeurIPS 2025, arXiv v2): abstract, Sec. 1-4 (Table 1-2, subnetwork-only fine-tuning), Sec. 6 incl. footnote on numerical precision"
digest_buckets: [1-missing-method, 4-supports]
importance: 4
---

# Reinforcement Learning Finetunes Small Subnetworks in Large Language Models (Mukherjee 2025)

**What it does.** Diffs released checkpoints before and after RL (DPO, KTO, SimPO, PPO, GRPO, PRIME and R1-Zero, across
10 models including Tulu 8B/70B, Eurus 7B, DeepSeek-Math 7B) and before and after SFT. It counts parameters left
unchanged, treating bf16 values within 1e-5 as equal. It then checks the rank of each update, the overlap of the updated
sets across seeds, data and algorithms, and whether fine-tuning only that subnetwork reproduces full RL.

**Key result.** RL leaves 68.5% to 96.0% of parameters unchanged (R1-Zero 86.0%), while SFT stages leave only 6-15%
unchanged. Updates touch nearly all matrices (layer norms almost never) and are nearly full rank (99.2-99.8% average
rank). Fine-tuning only the subnetwork reproduces the full model: 94.0% (DPO) and 90.5% (PRIME) of weights are
identical, and 100% at tolerance 1e-4. The subnetworks overlap across seeds and algorithms more than chance. The authors
attribute the sparsity to training on near-policy data, not to the KL penalty or clipping, and say the sparsity reflects
updates too small for floating point (footnote, Sec. 6).

**Quote.** "RL updates a small subset of parameters that nevertheless span almost the full subspaces that the parameter
matrices can represent." (abstract)

**Relevance to our paper.** This is weight-level diffing of real RL and SFT fine-tunes, so it bears on our Qwen RL/SFT
appendix and on our "effective rank" and update-size readings. Its finding that RL updates are full rank fits our
finding that trained updates are not low-rank rescalings of existing parts. Its sparsity is partly a bf16-precision
artifact, which the authors concede and which matches our own bf16-rounding lesson: any sparsity or size reading we
make on Qwen updates must be in fp32 or state its tolerance. It says nothing about mechanisms, so it is not a
known-answer case. The cheap baseline it suggests, the fraction of weights changed and the subnetwork overlap, could run
on our fine-tunes for $0 but would only measure size, which we already show tells update from noise and not reuse from
new. Worth one sentence in related work on RL vs SFT ("RL makes small, sparse updates", alongside Yue et al.).

**For the digest.**
- 1-missing-method: update sparsity and subnetwork overlap between checkpoints: trivially runnable on our 67M and Qwen pairs, but measures size only.
- 4-supports: RL updates are sparse but full rank, i.e. not a low-rank re-weighting; this fits "trained fine-tunes turn rather than scale".
