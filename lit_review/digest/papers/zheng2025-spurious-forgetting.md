---
slug: zheng2025-spurious-forgetting
title: Spurious Forgetting in Continual Learning of Language Models
authors: Zheng, Cai, Qiu, Ma
year: 2025
id: 2501.13453
tags: [EH, SUPPORTS, DYNAMICS, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: arXiv HTML abstract, secs 1-4 and 5.1 (motivation, synthetic study, weight and feature analysis, theory, method comparison)
digest_buckets: [4-supports]
importance: 4
---

# Spurious Forgetting in Continual Learning of Language Models (Zheng 2025)

**What it does.** Argues that many performance drops in continual fine-tuning are losses of "task alignment", not of
knowledge. Main evidence is a controlled study: a model pretrained from scratch on a synthetic 200k-person biography
corpus, then fine-tuned on QA for one set of people (Task 0) and then another (Task 1); knowledge retention is probed
by a recovery fine-tune on half of Task 0 and tested on the other half. Weight-update subspace angles (SVD of ΔW) and
principal-component shifts are measured, a linear residual-network theory is given, and freezing the bottom layers is
proposed. Also LLaMA-2-7B-Chat safety (10 identity-shift examples) and the TRACE continual-instruction benchmark.

**Key result.** Task 0 accuracy falls from ~100% to ~10% within the first 150 steps of Task 1, while recovery accuracy
stays ~100% then (96% at the end). The first 150 steps update weights in the same subspace as Task 0's update (undoing
it), and later steps in a near-orthogonal subspace, mostly in the bottom layers and embeddings. Freezing bottom layers
raises sequential Task 0 accuracy from 11% to 44% (others peak at 22%). On LLaMA-2-7B-Chat, safety drops from 100% to 0%
after 10 identity-shift examples and returns to ~99% after 10 refusal examples.

**Quote.** "performance drops often reflect a decline in task alignment rather than knowledge loss" (Abstract)

**Relevance to our paper.** Behavioural (recovery-probe) support for the view that fine-tuning mainly moves a small,
cheaply reversible alignment component on top of retained knowledge, the elicitation picture in continual-learning
form; its recovery probe is the same kind of test as Jain et al.'s reverse fine-tuning, which we found fails its
controls. The weight analysis is closer to us: alignment updates live in near-orthogonal subspaces and concentrate in
low layers, a weight-level observation we could check on our trained fine-tunes. It gives no mechanism-level reuse
evidence, and the synthetic setup (knowledge vs QA format) is a possible known-answer construction rather than one
read here. Not to be confused with shao2025spurious (Spurious Rewards), which the draft cites.

**For the digest.**
- 4-supports: forgetting is mostly loss of a reversible "task alignment" while knowledge stays recoverable; behavioural support for EH, with a weight-subspace analysis we could mirror.
