---
slug: hong2024-concept-vectors-mlp
title: Intrinsic Test of Unlearning Using Parametric Knowledge Traces
authors: Yihuai Hong et al.
year: 2024
id: 2406.11614
tags: [KNOWN-ANSWER, FT-INTERP, DIFFING, ALIGNMENT, SUPPORTS]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, baselines]
read_from: arXiv HTML abstract, sections 1, 3-6 (method, ConceptVectors benchmark, Table 1 results, jailbreak analysis)
digest_buckets: [1-missing-method, 2-known-answer-case, 4-supports]
importance: 5
---

# Intrinsic Test of Unlearning Using Parametric Knowledge Traces (Hong 2024)

**What it does.** Locates "concept vectors" in LLaMA2-7B-chat and OLMo-7B: single MLP value vectors (rows of the
second MLP matrix) whose vocabulary projection is dominated by one concept (e.g. Harry Potter), validated by ablation.
The ConceptVectors benchmark (285 concepts) then measures, after each unlearning fine-tune (gradient ascent, gradient
difference, DPO, NPO, NPO+KL, MEMIT, RMU), how far the concept vector itself moved (Jaccard of top projected tokens,
cosine, L2) alongside behavioural QA. An oracle "Needle" ablates the concept vector directly. It also measures concept
vector activations under jailbreaks of the unlearned models.

**Key result.** Gradient- and preference-based unlearning drive target QA and completion below 0.25 while leaving the
concept vector almost unchanged (Jaccard > 0.98; e.g. RMU 0.999). Needle drops Jaccard below 0.05 and cuts target QA
40-60% while keeping unrelated QA; ablating a random vector leaves target QA near 1. Jailbreaks recover the knowledge
by raising the concept vectors' activations, and ablating the vector reduces jailbreak success.

**Quote.** "existing unlearning methods suppress the parametric knowledge rather than erase it" (Section 1)

**Relevance to our paper.** A realistic fine-tune with independent internal evidence of what it did: the unlearning
fine-tunes keep the base mechanism's weights (the concept vector, read in weights) and change how strongly it is
used (its activation, recovered by jailbreak), with causal backing from Needle ablation. That is a known-answer case of
"same write, changed use", the case our re-gating share is built for, and it has a ready answer per concept on 7B
models. In our definition's terms the concept vector would read as kept (its write is unchanged up to scale) and the
change sits in its parents, so the behaviour is carried by reuse; worth checking that our readings say so. Its
per-vector weight diff (vocabulary-projection Jaccard) is also a simple weight-level diffing reading that our
baselines lack, though it only applies where a component's write is readable in vocabulary space. Supports the picture
that fine-tunes (here, removal ones) change gating more than writes, the reverse of what our trained 67M fine-tunes
show.

**For the digest.**
- 1-missing-method: per-component weight diff of MLP value vectors read through vocabulary projection; cheap, applies to Qwen3-1.7B, not tested against built answers.
- 2-known-answer-case: LLaMA2-7B-chat / OLMo-7B, unlearning fine-tunes (GA, DPO, NPO, RMU) on a concept: concept vector kept (Jaccard > 0.98), activation suppressed (jailbreak raises it), Needle ablation shows the vector is causal.
- 4-supports: fine-tuning that changes behaviour leaves the base component's write intact and changes its use.
