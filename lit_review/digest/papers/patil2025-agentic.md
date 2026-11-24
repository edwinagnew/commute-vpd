---
slug: patil2025-agentic
title: "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models"
authors: Shishir G. Patil et al.
year: 2025
id: https://proceedings.mlr.press/v267/patil25a.html
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "PMLR PDF pages 1-8 (abstract, intro, dataset and evaluation sections, sec. 5.1 results text); identified from Oh et al. 2026 (2610.01509) reference list"
digest_buckets: []
importance: 1
---

# The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models (Patil 2025)

**What it does.** A benchmark for function calling: serial and parallel calls in several languages, crowd-sourced
functions, hallucination (abstention) tests, and multi-turn, stateful agentic scenarios. Calls are scored by an Abstract
Syntax Tree matcher, which the authors check against executed calls, and multi-turn tasks by state and response checks.

**Key result.** Top models do well on single-turn, crowd-sourced and hallucination categories. Multi-turn and agentic
tasks, memory management above all, leave "significant room for improvement". BFCL has become the de facto standard
for function-call evaluation.

**Quote.** "while state-of-the-art LLMs excel at single-turn calls, memory, dynamic decision-making, and long-horizon reasoning remain open challenges." (Abstract)

**Relevance to our paper.** None. Oh et al. cite it as one of the benchmarks for asking whether post-training sharpens
agentic behaviour; it is an evaluation suite, not a study of what fine-tuning changes.

**For the digest.**
