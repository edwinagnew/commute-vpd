---
slug: schick2023-agentic
title: "Toolformer: Language Models Can Teach Themselves to Use Tools"
authors: Timo Schick et al.
year: 2023
id: 2302.04761
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, sec. 2 approach, sec. 4 experiments incl. 4.4 scaling laws"
digest_buckets: []
importance: 2
---

# Toolformer: Language Models Can Teach Themselves to Use Tools (Schick 2023)

**What it does.** GPT-J (6.7B) annotates a subset of CCNet with candidate API calls (calculator, QA system, Wikipedia
search, translation, calendar) from a few demonstrations. It keeps a call only if its result lowers the loss on the
following tokens, then fine-tunes on the annotated text. The scaling study repeats this on GPT-2 models from 124M to
1.6B.

**Key result.** Toolformer improves zero-shot results on LAMA, maths, QA and temporal tasks, often matching much larger
models, without hurting language modelling. The tool-use ability "only emerges at around 775M parameters": below that,
models score about the same with and without tools.

**Quote.** "the ability to leverage the provided tools only emerges at around 775M parameters: smaller models achieve similar performance both with and without tools." (Sec. 4.4)

**Relevance to our paper.** Mostly background. Oh et al. cite it for agents calling diverse tools. The scale threshold
is a small behavioural point for elicitation, since the same fine-tune teaches tool use only when the base is large
enough. It is a reason a 67M model may not show elicitation the way large models do, though the paper does not test
mechanisms.

**For the digest.**
