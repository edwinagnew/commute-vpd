---
slug: deeb2024-unlearning-relearn
title: Do Unlearning Methods Remove Information from Language Model Weights?
authors: Aghyad Deeb et al.
year: 2024
id: 2410.08827
tags: [ALIGNMENT, EH, SUPPORTS]
primary_tag: ALIGNMENT
in_our_bib: deeb2024unlearning
cited_in_draft: yes
themes: [elicitation, other]
read_from: arXiv HTML abstract, sections 1-3 (method), section 4 setup; results via abstract/contributions and Figure 1 text
digest_buckets: [4-supports]
importance: 3
---

# Do Unlearning Methods Remove Information from Language Model Weights? (Deeb 2024)

**What it does.** Proposes an adversarial test of whether unlearning removes information from weights or only hides
it. Facts that cannot be guessed from each other (e.g. dates of events) are split into T and V; both are unlearned;
the attacker fine-tunes on T only and measures accuracy on V. Because T and V are independent, recovering V cannot be
re-teaching. Main model Llama 3 8B; methods include gradient ascent, RMU and training on incorrect facts.

**Key result.** Fine-tuning on the accessible facts "can recover 88% of the pre-unlearning accuracy" for every method
tested when the knowledge was learned in pretraining and the unlearning keeps other tasks intact. Unlearning knowledge
that was added by an extra fine-tuning phase looks more robust than unlearning pretraining knowledge, so evaluations
of that kind may overestimate robustness.

**Quote.** "using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to
current unlearning methods for information learned during pretraining" (Abstract)

**Relevance to our paper.** Already cited in the draft's "why mechanisms" notes for "unlearning that fine-tuning
undoes", which matches what it says. Its contrast between pretraining-learned and fine-tune-learned knowledge is a
small additional point for us: knowledge put in by a fine-tune behaves differently under later fine-tuning than
pretraining knowledge, consistent with a fine-tune building something shallower than pretraining did. It is purely
behavioural (the paper notes probes failed to detect what relearning recovers), so it is background for the
mechanism argument, not a baseline or known answer.

**For the digest.**
- 4-supports: a behaviour change (unlearning) that independent-fact relearning shows left the capability in the weights; supports reading reuse inside the model rather than from behaviour.
