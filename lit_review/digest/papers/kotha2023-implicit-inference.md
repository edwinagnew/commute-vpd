---
slug: kotha2023-implicit-inference
title: Understanding Catastrophic Forgetting in Language Models via Implicit Inference
authors: Kotha, Springer, Raghunathan
year: 2023
id: 2309.10105
tags: [EH, DEFINITION, SUPPORTS, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: arXiv HTML abstract, secs 1-4 (synthetic regression setup, hypothesis, language-model experiments)
digest_buckets: [3-definition, 4-supports]
importance: 5
---

# Understanding Catastrophic Forgetting in Language Models via Implicit Inference (Kotha 2023)

**What it does.** In a synthetic setup, a 22.4M GPT-2-style transformer is pretrained to in-context-learn linear
regression on a mixture of a Gaussian task prior and 64 fixed tasks, then fine-tuned on the fixed tasks. The authors
factor the model's output into "capabilities" (ridge vs discrete regression) weighted by "task inference" (the
posterior on which task the prompt comes from), and test it with "conjugate prompting": a transformed prompt that
looks less like the fine-tuning data but has an invertible answer. On LLMs the transform is translation (plus
Leetspeak and Pig Latin), on LLaMA vs Alpaca/Vicuna, OPT vs OPT-IML, Llama-2 vs Code Llama, and GPT-3.5 before and after
safety tuning.

**Key result.** Fine-tuning raises loss most on prompts near the fine-tuning distribution and less on far ones;
rescaling the prompt recovers ridge-regression performance, so the authors call the capability "suppressed" rather
than forgotten. In LLMs, the instruction-tuned models' drop in in-context learning is almost always smaller in
non-English prompts (Alpaca: 1.5% drop in Leetspeak and 13.5% in Pig Latin vs 56.75% in English). Code Llama's English
XNLI loss mostly vanishes in French, Spanish and German, and harmful-answer rates drop less after RLHF in
non-English languages.

**Quote.** "Due to limited mechanistic understanding, we can not test whether this is how language models compute
solutions." (Sec 2.7)

**Relevance to our paper.** A behavioural argument for the elicitation hypothesis in its forgetting form: fine-tuning
"skews model task inference more than it changes models capabilities" (Intro). Its capability-times-task-inference
factorization is a definition to contrast with ours: it splits a model into reused capabilities and a re-weighting
by input, which is close to our re-gating reading, but it is assumed, not measured, and the quote says so. That is the
gap our paper claims to fill, so the draft should cite it in "post-training as selection" with Jain et al.'s revival
tests. There is no mechanistic known-answer case (the synthetic model's internals are never examined), but the
regression setup with a known ridge/discrete mixture could in principle be one.

**For the digest.**
- 3-definition: capability vs task inference (a mixture weight over retained solutions); a behavioural, assumed form of "re-weighting" to contrast with our mechanism-level reuse.
- 4-supports: fine-tuning losses are largely recoverable by moving the prompt off the fine-tuning distribution, behavioural support for EH-as-reweighting.
