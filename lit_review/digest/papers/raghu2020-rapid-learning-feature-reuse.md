---
slug: raghu2020-rapid-learning-feature-reuse
title: "Rapid learning or feature reuse? Towards understanding the effectiveness of MAML"
authors: Raghu et al.
year: 2020
id: 1909.09157
tags: [DEFINITION, BACKGROUND, CITE-FIX]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, other]
read_from: "local PDF text (~/commute-data/lit_review_pdfs/survey/1909.09157), pp. 1-9: abstract, Sec. 3 (freezing, CCA/CKA), Sec. 4-5 (ANIL, NIL tables); appendix not read"
digest_buckets: [3-definition, 5-cite-fix]
importance: 4
---

# Rapid learning or feature reuse? Towards understanding the effectiveness of MAML (Raghu 2020)

**What it does.** Asks whether MAML's inner-loop adaptation works by "rapid learning" (large changes to the body's
representations) or by "feature reuse" (the meta-initialization already has the features). Setting: the standard
4-conv MAML network on MiniImageNet and Omniglot few-shot tasks, plus RL benchmarks. Two tests: freeze body layers
during test-time adaptation and read accuracy, and compare each layer's activations before vs after adaptation by
CCA/CKA. This leads to ANIL (adapt only the head) and NIL (no head, cosine similarity on features).

**Key result.** Freezing all four conv layers at test time barely moves accuracy (MiniImageNet 5-way 1-shot 46.9 to
46.3; 5-shot 63.1 to 61.0, Table 1). Body layers have CCA similarity above 0.9 before vs after adaptation, the head
below 0.5 (Fig. 2). ANIL matches MAML (e.g. 46.7 vs 46.9, 61.5 vs 63.1 on MiniImageNet), and the pattern holds from
early in meta-training.

**Quote.** "In feature reuse, the meta-initialization already contains highly useful features that can mostly be
reused as is for new tasks, so little task-specific adaptation occurs." (Sec. 3)

**Relevance to our paper.** The best-known ML paper that uses "feature reuse" as a named hypothesis against "learning
new", so a natural citation where we introduce our definition of reuse. Its reuse is read by value, not mechanism:
a layer is reused if freezing it costs no accuracy or its activations stay CCA-similar, and CCA is invariant to
invertible linear maps, so a rotated layer would read as reuse. That is the opposite design choice to our "kept up to
a positive scalar" clause, and a useful contrast now that a rotation-tolerant kept clause is pending: Raghu et al.'s
reading is the extreme rotation-tolerant end. It is meta-learning on small CNNs with a few inner steps, so it is no
known-answer case for language-model fine-tunes, and the freezing test is close in spirit to Jain et al.'s tests that
we already adapt.

**For the digest.**
- 3-definition: earliest operational "feature reuse vs rapid learning" split (freeze test + CCA/CKA similarity); cite
  when defining reuse and contrast: by value and invariant to rotation, where ours is by mechanism and up to a scalar.
