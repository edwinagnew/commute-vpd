---
slug: turnertrout2025-self-fulfilling-misalignment
title: Self-fulfilling misalignment data might be poisoning our AI models
authors: Turner (TurnTrout)
year: 2025
id: https://www.lesswrong.com/posts/QkEyry3Mqo8umbhoK/self-fulfilling-misalignment-data-might-be-poisoning-our-ai
tags: [ALIGNMENT, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: "LessWrong linkpost page (curl) plus the full post at turntrout.com/self-fulfilling-misalignment via WebFetch summary"
digest_buckets: []
importance: 2
---

# Self-fulfilling misalignment data might be poisoning our AI models (Turner 2025)

**What it does.** A position post. It hypothesizes that pretraining text predicting misaligned AI makes models more likely to adopt misaligned goals. The evidence it collects is out-of-context reasoning (Berglund et al.), emergent misalignment (Betley et al.), training on reward-hacking documents, influence functions tracing a shutdown answer to HAL-9000 fiction (Grosse et al.), and alignment faking. It then proposes tests and mitigations: filtering, upsampling positive AI text, conditional pretraining and gradient routing.

**Key result.** No new experiments. It prompted the controlled study of Tice et al. 2026 (Alignment Pretraining).

**Quote.** "Your AI's training data might make it more 'evil' and more able to circumvent your security, monitoring, and control measures." (opening)

**Relevance to our paper.** This is the origin of the "alignment pretraining" thread our related work plans to cover. It frames dispositions as installed in pretraining and later activated, which is the elicitation view applied to personas. Turner hedges that the relevant persona may be "only woken by several thousand finetuning examples". If the paragraph is written, cite Tice et al. 2026 for the evidence and this post at most as the origin of the idea.

**For the digest.** (no bucket)
