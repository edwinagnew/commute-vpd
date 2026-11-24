---
slug: xu2025-unlearning-activation-geometry
title: "Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs"
authors: Xiaoyu Xu et al.
year: 2025
id: 2505.16831
tags: [ALIGNMENT, DIFFING, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, other]
read_from: arXiv HTML abstract, sections 1-3, section 5.1 (perturbation model), model list in section 4
digest_buckets: []
importance: 2
---

# Unlearning Isn't Deletion (Xu 2025)

**What it does.** Studies whether unlearning (GA, GA+GD, GA+KL, NPO, single and continual up to 100 requests) on
Yi-6B, Qwen2.5-7B and several Llama-3 / Qwen3 models can be reversed by relearning, and reads the change with
representation-level tools: PCA subspace similarity and shift, CKA between layer activations, Fisher information,
and a summary "mean PCA distance". It sorts outcomes into four regimes by reversibility and catastrophicity.

**Key result.** Task metrics (forget accuracy, perplexity, MIA) cannot tell reversible from irreversible forgetting:
accuracy can collapse while representations stay intact and recoverable by brief relearning, even on unrelated data.
Small, localized perturbations (it argues, near the logits) distort task metrics with features intact; large,
distributed ones cause irreversible drift. Irreversible, non-catastrophic forgetting is rare.

**Quote.** "Small perturbations near the logits can distort task-level metrics despite intact features, hence leading
to misleading assessments." (Section 1, contributions)

**Relevance to our paper.** Background for the argument that behaviour misleads about what changed inside. Its
diagnostics (CKA, PCA shift) are global activation-similarity readings; they say how much the representation moved,
not whether a mechanism is reused, so they are weaker than our existing baselines and not worth adding. The
perturbation-model argument is heuristic. Little direct use.

**For the digest.** (none)
