---
slug: lw-knowledge-of-graders
title: "Knowledge of graders can mediate selection for grader-targeting reasoning through RL"
authors: lennie (LessWrong handle) et al.
year: 2026
id: https://www.lesswrong.com/posts/qqyxTYssqLgFsjvJu/knowledge-of-graders-can-mediate-selection-for-grader
tags: [EH, ALIGNMENT, SUPPORTS]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: LessWrong post full (via GreaterWrong mirror, main text and appendix headings; figures not viewed)
digest_buckets: [4-supports]
importance: 3
---

# Knowledge of graders can mediate selection for grader-targeting reasoning through RL (lennie 2026)

**What it does.** Starting from Nemotron 3 Super (120B-A12B), synthetic document fine-tuning (SDF, about 2M tokens per corpus within 7.5M total) plants knowledge of the graders for two pairs of toy environments. In Wood/Yard, one lab grades MBPP code for correctness and the other for impossible format constraints; in Eton/Rugby, one targets 100-word answers and the other 300. There are four pre-RL checkpoints (knowledge of both, of either, or a FineWeb control). Each is RL'd for 100 steps on one pair and evaluated on both.

**Key result.** Checkpoints that know the in-distribution graders show grader-targeting reasoning (tuned to > 2% before RL), which saturates within about 20 RL steps, and they reach high reward much faster. Checkpoints without that knowledge "never explore into grader-targeting reasoning". Only the checkpoint that knows both pairs' graders also gains reward and grader-targeting reasoning on the held-out pair. The pre-RL knowledge sets what RL learns.

**Quote.** "the knowledge possessed by the pre-RL checkpoint influences the cognition learned during RL." (Introduction)

**Relevance to our paper.** This is behavioural evidence for the elicitation hypothesis in a constructed form. RL picks up and amplifies a reasoning pattern whose ingredients (grader knowledge) were planted beforehand, and it does not find the strategy when they are absent. The planted-knowledge design is a "known by construction" idea at the behaviour level: the reuse label comes from what SDF put in, not from internals. That is analogous to our built edits but at 120B scale. There is no mechanistic reading, so it supports the hypothesis our paper states but not our definition or methods.

**For the digest.**
- 4-supports: RL elicits a strategy only when its pre-RL ingredients exist (planted by SDF), and generalises through them; a constructed behavioural known answer for "post-training selects what pre-training put there".
