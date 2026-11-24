---
slug: arnold2025-em-phase-transitions
title: "Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment"
authors: Arnold et al.
year: 2025
id: 2508.20015
tags: [ALIGNMENT, DYNAMICS, DIFFING]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, model-diffing]
read_from: arXiv HTML abstract, sections 1-3.3, conclusion and outlook (numbers cross-checked against the old summary because the HTML math rendering doubled digits)
digest_buckets: []
importance: 2
---

# Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment (Arnold 2025)

**What it does.** Tracks the output distribution of Qwen2.5-14B-Instruct across checkpoints of a rank-1 LoRA on one
MLP down-projection (layer 24 of 48), trained one epoch on bad medical advice (the Turner et al. emergent-misalignment
recipe). A "linear dissimilarity" (an f-divergence between output distributions of neighbouring checkpoints) detects
the transition; LLM-judged "order parameters" (alignment, verbosity, formatting, confidence, ...) coarse-grain the
outputs, and by the data-processing inequality the ratio of coarse to full dissimilarity is the share of the change
each order parameter explains.

**Key result.** The behavioural transition comes later than the gradient-norm peak (step 59 of 397), which is an early
warning rather than the transition. The alignment order parameter explains only about 3% of the total output change;
six stylistic ones each explain single-digit percentages, and together with alignment about 27%. Misalignment ends at
26-35% of responses.

**Quote.** "emergent misalignment represents only a fraction of the coordinated behavioral shift that occurs during fine-tuning." (§5)

**Relevance to our paper.** Purely behavioural, so it does not test reuse. Two small points of contact: the "share of
the total change explained" construction resembles our share readings (a fraction of a total change carried by a
named part), though in output space; and it is a reminder that a narrow fine-tune changes much more than the target
behaviour, which matters when our readings average over a whole update. The rank-1 single-layer LoRA is a clean
fine-tune to diff, but the paper gives no internal evidence of reuse or novelty, so it is not a known-answer case.
Low priority.

**For the digest.** (none)
