---
slug: turner2023-actadd
title: "Steering Language Models With Activation Engineering"
authors: Turner et al.
year: 2023
id: 2308.10248
tags: [BASELINE, DEFINITION, EH]
primary_tag: BASELINE
in_our_bib: turner2023
cited_in_draft: yes
themes: [baselines, elicitation]
read_from: "arXiv HTML: abstract, intro, section 3 (method), section 4 results (4.1-4.3), discussion and limitations"
digest_buckets: [3-definition]
importance: 4
---

# Steering Language Models With Activation Engineering (Turner 2023)

**What it does.** Introduces Activation Addition (ActAdd): run the model on a contrast pair of prompts (e.g. "Love"
vs "Hate"), take the difference of residual-stream activations at one layer, scale it by a coefficient, and add it
to the forward pass at inference. No optimisation; one prompt pair. Tested on GPT-2-XL (topic steering, perplexity on
OpenWebText), OPT-6.7B and LLaMA-3-8B (RealToxicityPrompts, sentiment shift).

**Key result.** The "wedding" vector lowers perplexity on wedding-related sentences while leaving unrelated text
about unchanged, and at the best layer yields wedding words in over 90% of completions against a ~2% baseline.
ActAdd-OPT is 8% less toxic than the second-best method (PREADD-D-OPT); ActAdd-LLaMA-3 gives a 5% drop, with a small
fluency cost. The authors frame the motivation as an "elicitation overhang".

**Quote.** "we doubt that activation engineering can e.g. teach a model a new skill" (Section 5, "Activation
engineering vs finetuning")

**Relevance to our paper.** The draft cites it correctly as the source of activation addition and says our steering
share differs in taking the difference between two models on the same inputs rather than between contrasting
prompts. The paper's own framing supplies the rationale for reading a steering share as elicitation: steering only
adds a direction to the base computation, so what it recovers is what the base model can already do, and the authors
say it likely cannot teach new skills. That is the assumption our steering share rests on, and our built pairs (where
steering sides with the mechanism once prior and mechanism are pulled apart) test it; citing that sentence would make
the assumption explicit. "Elicitation overhang" is an informal, behavioural sense of elicitation that could be
contrasted with ours in one clause.

**For the digest.**
- 3-definition: "elicitation overhang" and the claim that steering cannot teach new skills are an informal, behavioural notion of elicitation to contrast with our mechanism-level one.
