---
slug: burns2023-weak-to-strong
title: "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision"
authors: Burns et al.
year: 2023
id: 2312.09390
tags: [EH, BACKGROUND, CITE-FIX]
primary_tag: EH
in_our_bib: burns2023weak
cited_in_draft: in refs.bib but not \cite'd
themes: [elicitation]
read_from: "arXiv HTML: abstract, intro, sec. 3 (PGR), sec. 4 results, sec. 5.2 saliency incl. 5.2.3 linear probes"
digest_buckets: [4-supports]
importance: 3
---

# Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (Burns 2023)

**What it does.** Fine-tunes base models of the GPT-4 family (spanning about 7 orders of magnitude of pre-training
compute) on labels produced by a weaker fine-tuned model, on 22 NLP classification tasks, chess puzzles and ChatGPT
reward modelling. It scores the result by the performance gap recovered (PGR): the share of the weak-to-ceiling gap the
weakly supervised strong model closes. It tests fixes (auxiliary confidence loss, bootstrapping, generative fine-tuning)
and probes how "salient" the task is in the strong model (prompting, linear probes).

**Key result.** PGR is almost always positive; for the largest students on NLP it is often above 50%, and the
confidence loss raises median PGR from about 25% to nearly 80% at the largest gap. Reward modelling recovers only about
10% (rarely above 20%). Few-shot prompting becomes competitive with fine-tuning at the largest sizes. Linear probes on
the base model with ground-truth labels reach 72% on average against 82% for full fine-tuning; fine-tuning on weak
labels then probing with true labels reaches 78%, so fine-tuning makes the concept more linearly represented.

**Quote.** "we do not need the weak supervisor to teach the strong model new capabilities; instead, we simply need the
weak supervisor to elicit what the strong model already knows." (sec. 1)

**Relevance to our paper.** A standard source for the word "elicit" in the post-training-as-selection sense, and for
the behavioural argument that fine-tuning mostly surfaces latent pre-trained knowledge (weak labels recover strong
performance; prompting matches fine-tuning at scale). Its evidence is entirely behavioural, so it is an example of the
gap our paper names ("evidence so far is behavioural"). The probe result is a small point for us: even on elicited
tasks the fine-tune changes the final representation (the concept becomes more linear), which fits our finding that
trained fine-tunes turn writes rather than only rescale. No known-answer use: no mechanism is traced. It is in refs.bib
but cited nowhere; either cite it in the "post-training as selection" paragraph or drop the entry.

**For the digest.**
- 4-supports: behavioural support for elicitation (weak supervision recovers much of a strong model's latent ability), with a hint that fine-tuning still reshapes representations.
