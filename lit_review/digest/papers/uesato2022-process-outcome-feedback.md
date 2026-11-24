---
slug: uesato2022-process-outcome-feedback
title: "Solving math word problems with process- and outcome-based feedback"
authors: Jonathan Uesato et al.
year: 2022
id: 2211.14275
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, sec. 2 (methods: SFT, ORM/PRM reward models, expert-iteration RL), sec. 3 result headings and text"
digest_buckets: []
importance: 2
---

# Solving math word problems with process- and outcome-based feedback (Uesato 2022)

**What it does.** Compares outcome-based supervision (final answer only) with process-based supervision (step-level
human labels) on GSM8K, with a 70B Chinchilla-class base model. Training uses few-shot prompting, SFT, outcome and process
reward models (ORM, PRM) for reranking, and RL by expert iteration against final-answer correctness or a reward model.

**Key result.** Outcome-only supervision reaches similar final-answer error with fewer labels. Low reasoning-trace
error needs process feedback, or a reward model that emulates it, and ORMs trained on outcomes turn out to approximate
PRM labels. The best results go from 16.8% to 12.7% final-answer error and from 14.0% to 3.4% trace error. RL helps
few-shot models a lot but gives "more modest gains on top of SFT".

**Quote.** "We find that pure outcome-based supervision produces similar final-answer error rates with less label supervision." (Abstract)

**Relevance to our paper.** Background only, as Oh et al. 2026 use it: a citation for RL on maths and for outcome
rewards rewarding flawed reasoning. The modest gain of RL over SFT is a weak behavioural point at most. It has no
internals, no diffing and no definition. It is little use to us.

**For the digest.**
