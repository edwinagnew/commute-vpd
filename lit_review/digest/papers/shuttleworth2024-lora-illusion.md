---
slug: shuttleworth2024-lora-illusion
title: "LoRA vs Full Fine-tuning: An Illusion of Equivalence"
authors: Shuttleworth et al. (Andreas, Torralba, Sharma)
year: 2024
id: 2410.21228
tags: [DIFFING, FT-INTERP, BASELINE, SUPPORTS, CITE-FIX]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes, baselines]
read_from: "local PDF text (NeurIPS 2025, arXiv v3): abstract, Sec. 1-4 (Def. 3.1, Algorithm 1, Fig. 4 results, forgetting and scaling interventions), Sec. 5 continual learning (skimmed)"
digest_buckets: [1-missing-method, 4-supports, 5-cite-fix]
importance: 5
---

# LoRA vs Full Fine-tuning: An Illusion of Equivalence (Shuttleworth 2024)

**What it does.** Compares the SVD of pre-trained and fine-tuned weight matrices for LoRA and full fine-tuning of
LLaMA2-7B (MetaMathQA, Magicoder) and RoBERTa-base (six classification tasks). It defines "intruder dimensions": top
singular vectors of the fine-tuned matrix whose maximum cosine with every pre-trained singular vector is below ε. It then
scales those singular values to test causally whether they cause forgetting, and runs continual learning.

**Key result.** LoRA adds intruder dimensions among the top 10 singular vectors, at ranks up to at least r = 256 on
LLaMA2-7B. Full fine-tuning "almost never" does, and keeps the pre-trained singular structure. Scaling the intruder
dimensions down improves pre-training loss with little loss on the task. LoRA forgets less than full fine-tuning at
equal task fit, and its forgetting is concentrated in the intruder dimensions. During continual learning intruder
dimensions accumulate and LoRA forgets more.

**Quote.** "weight matrices trained with LoRA have new, high-ranking singular vectors, which we call intruder dimensions,
while those trained with full fine-tuning do not" (abstract)

**Relevance to our paper.** This is a weight-level reading of "new direction vs existing direction" in fine-tune
updates, close to our outside share and to the rotation-versus-rescale question. It is a cheap baseline we do not run:
an intruder-dimension count per matrix could be computed for free on all our 67M and Qwen fine-tunes, including the
LoRA split. It predicts that our LoRA fine-tunes read as more "new" than full ones at equal behaviour, a known-answer-like
contrast we could check. It supports the premise that two fine-tunes with one behaviour can differ inside ("illusion of
equivalence"), which is our Claim 1's motivating case, so the draft should cite it there. Note the tension: full
fine-tuning "retains most of the pre-training structure" in singular vectors, while our trained full fine-tunes turn
the writes of needed variables. The two can coexist (small turns stay above ε), but the draft should address it.

**For the digest.**
- 1-missing-method: intruder-dimension count (SVD cosine to pre-trained singular vectors, threshold ε): free to run on all our updates; a weight-level "new direction" baseline.
- 4-supports: LoRA and full fine-tunes with equal task behaviour differ structurally inside the weights; supports Claim 1's premise (same behaviour, different mechanism).
