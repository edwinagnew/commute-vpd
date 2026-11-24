---
slug: wang2025-finetuning-circuit-analysis
title: "Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis"
authors: Xu Wang et al. (Hu, Du, Cheng, Wang, Zou)
year: 2025
id: 2502.11812
tags: [FT-INTERP, DIFFING, SUPPORTS, CITE-FIX, CAUSAL]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "arXiv HTML v2 (ICML 2025): abstract, Sec. 1, 2.2, 3 (EAP-IG, faithfulness, robustness, tasks), 4 (circuit evolution), 5 (CircuitLoRA), 7; appendices not read; accuracy curves are in figures only"
digest_buckets: [4-supports, 1-missing-method, 5-cite-fix]
importance: 5
---

# Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis (Wang 2025)

**What it does.** The paper chooses math tasks on which the base model starts poorly, unlike Prakash et al. and Chhabra et
al.: add/sub, mul/div, sequences, LCM and linear functions. It fine-tunes Pythia-1.4B (LoRA, and full fine-tuning in an
appendix), with GPT-Neo-2.7B and OPT-6.7B as checks. Circuits are found with EAP-IG (top 5% of edges) at checkpoints along
training, and their node and edge sets are compared. A circuit-aware LoRA (more rank on layers whose edges change most) is
built from the result.

**Key result.** Circuits are faithful (mostly above 80%; 77.52% for add/sub before fine-tuning) in both the base and the
fine-tuned models, despite the large accuracy gap. Across checkpoints, the edge change rate is 2-3 times the node change
rate. Added nodes sit in middle and later layers, and edge changes concentrate in the middle layers. CircuitLoRA beats LoRA
and a random-layer LoRA by 2.46% on average.

**Quote.** "while circuits maintain high node similarity before and after fine-tuning, their edges undergo significant
changes, which is in contrast to the previous work (Prakash et al., 2024; Chhabra et al., 2024) that show circuits only
add some additional components after fine-tuning." (abstract)

**Relevance to our paper.** It is a direct counterpoint to Prakash's "enhances existing mechanisms" on fine-tunes where the
base model is poor, and the draft should cite it in the post-training-as-selection or diffing related work. Its reading
(same nodes, rewired edges) is close in spirit to our finding that trained fine-tunes keep the needed variables but change
their mechanisms ("turn writes rather than scale them"). It is not independent evidence of reuse or new: the circuits come
from an attribution threshold, and edge "changes" are threshold crossings with no ablation or patching check. So it is not
a known-answer case. As a baseline, an EAP-IG circuit diff between M and F (node and edge Jaccard) is cheap to run on the
67M fine-tunes and on our built pairs. It would test whether circuit-membership diffs track the construction, which our
neuron CMAP baseline (also attribution-ranked) suggests they may not.

**For the digest.**
- `4-supports`: on base-poor math fine-tunes, nodes are kept and edges rewired, which is consistent with our "mechanisms of kept variables change" reading (attribution evidence only).
- `1-missing-method`: an EAP-IG node/edge circuit diff between M and F is a runnable baseline on our built pairs.
