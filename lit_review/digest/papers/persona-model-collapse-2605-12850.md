---
slug: persona-model-collapse-2605-12850
title: Persona-Model Collapse in Emergent Misalignment
authors: Davi Bastos Costa, Renato Vicente
year: 2026
id: 2605.12850
tags: [ALIGNMENT, FT-INTERP, DIFFING]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing]
read_from: "arXiv HTML: abstract, method, KL and mechanistic sections (first ~100k chars); tail is checklist only"
digest_buckets: []
importance: 3
---

# Persona-Model Collapse in Emergent Misalignment (Bastos Costa 2026)

**What it does.** Proposes that emergent misalignment (EM) is "persona-model collapse": fine-tuning on narrow harmful data
degrades the model's ability to represent and tell apart personas. The models are GPT-4.1, GPT-4o, Qwen3-235B-A22B and
Qwen3.5-397B, each fine-tuned on insecure code, with secure code as the benign control. Each model role-plays 100 personas
answering the Moral Foundations Questionnaire, and the authors score the spread across personas and the stability within
one. For Qwen3-235B they also measure the participation ratio of last-prompt-token residual states over 1,000 persona
prompts at every layer, a count of how many directions carry the variance.

**Key result.** Answer distributions move 2.03 bits (KL) from base after insecure fine-tuning, against 0.57 for secure and
0.19 base-to-base. In GPT-4o, the share of extreme ratings rises from 19% to 73%. In the last layer the insecure variant's
participation ratio is 10.5% below base, about 7 points more than the secure variant's drop. Averaged over layers, the gap
is small (-0.52 points). Each condition was trained once.

**Quote.** persona-model collapse is "a deterioration of the internal representations that encode and distinguish personas" (abstract)

**Relevance to our paper.** It is a base-versus-fine-tune activation comparison, but only a geometry statistic, and a weak
effect from one run per condition. It cannot say whether a base mechanism is reused or replaced. Its framing, that a
fine-tune degrades an existing persona machinery, is a third option beside "reuse" and "new" that our definition files
under "not kept". Low value to us.

**For the digest.**
