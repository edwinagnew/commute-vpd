---
slug: zhang2025-pretrain-midtrain-rl
title: "On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"
authors: Zhang et al.
year: 2025
id: 2512.07783
tags: [EH, KNOWN-ANSWER, SUPPORTS, CONTRADICTS]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "local PDF arxiv_2512.07783.pdf (v1) via pdf-text: abstract, §1-2.4 (framework, training setup), Observations/Takeaways 1-5, §8 conclusion; appendices not read; figure values only where stated in text"
digest_buckets: [2-known-answer-case, 4-supports, 4-contradicts]
importance: 5
---

# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models (Zhang 2025)

**What it does.** Trains 100M-parameter Qwen2.5-style models from scratch on 10B tokens of synthetic GSM-Infinite
problems. Each problem is a dependency graph of arithmetic operations, rendered in a chosen context template such as
zoo animals or teachers. The pre-training, mid-training and GRPO post-training data are disjoint. The paper varies how
hard the RL data is relative to pre-training (op counts), how much of a second context appears in pre-training (0% to
10%), and the compute split between mid-training and RL. It scores answers only when the parsed reasoning graph is also
correct.

**Key result.** RL on in-distribution difficulties (op 2-10) raises pass@1 but not pass@128: it sharpens. RL at the
"edge of competence" (op 11-14) raises pass@128 on harder out-of-distribution problems, and RL on tasks far too hard
fails. Transfer to context B fails when pre-training had 0% or 0.1% of B and works when B was 1% or more (up to +60%
pass@128). Mid-training plus RL beats RL alone at fixed compute (+10.8% on OOD-hard). Process rewards reduce reward
hacking.

**Quote.** "RL incentivizes contextual generalization only when the relevant primitives or base skills are present in
the base model." (§1)

**Relevance to our paper.** This is a ready-made known-answer generator at our scale: a 100M model whose pre-training
content is fully controlled. Running the same RL from two bases that differ only in whether context B appeared in
pre-training (0% vs 1%) gives a pair where the primitive is absent or present by construction. Running RL on op 7-10
vs op 11-14 gives a sharpen-only update and a capability-extending one on the same model. Both are natural tests for
whether ρ and the re-gating share follow the construction on trained (not built) updates, which is exactly where our
readings fail now. The evidence is behavioural (pass@k with a process check). The paper says "causal mechanisms" only
loosely, and there is no activation or weight analysis. The behavioural reading supports EH in-distribution and goes
against it at the edge of competence.

**For the digest.**
- 2-known-answer-case: 100M from-scratch models with controlled exposure (0% vs ≥1% context B; ID vs edge-of-competence RL data) give constructed RL pairs. This is the closest scale match to our 67M in this batch; we would need to train it and fit a decomposition.
- 4-supports: RL on data the base already covers only sharpens (pass@1 up, pass@128 flat).
- 4-contradicts: RL at the edge of competence extends pass@128, so post-training is not only re-weighting there.
