---
slug: tofu-benchmark
title: "TOFU: A Task of Fictitious Unlearning for LLMs"
authors: Maini et al.
year: 2024
id: 2401.06121
tags: [ALIGNMENT, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "identified from the name (TOFU, Maini et al.); arXiv abs page + arXiv HTML: intro, section 2 (dataset, model utility, forget quality)"
digest_buckets: []
importance: 1
---

# TOFU: A Task of Fictitious Unlearning for LLMs (Maini 2024)

**What it does.** Builds 200 GPT-4-written fictitious author profiles, 20 question-answer pairs each, fine-tunes
Llama-2-7B-chat and Phi-1.5 on all of them, and asks unlearning methods to forget a 1%, 5% or 10% subset. Forget
quality is a statistical test (KS on a truth-ratio statistic) against a "retain model" fine-tuned only on the other
profiles, the gold standard of never having learned the forget set; model utility aggregates several held-out scores.

**Key result.** None of the four baselines (gradient ascent, gradient difference, KL minimisation, preference
optimisation toward "I don't know") reaches the retain model's forget quality without losing utility.

**Quote.** "none of the baselines we consider show effective unlearning" (Abstract)

**Relevance to our paper.** Almost none. The facts are taught by a fine-tune first, so a base-model decomposition does
not cover them, as the alignment-applications note says. Its one transferable idea is the retain-model reference: a
model that never saw the data as the known answer for "removed", the unlearning analogue of our built known answers.

**For the digest.** (no buckets)
