---
slug: guo2024-localized-unlearning
title: "Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization"
authors: Phillip Guo et al.
year: 2024
id: 2410.12949
tags: [FT-INTERP, CAUSAL, ALIGNMENT, BASELINE]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, baselines, causal-abstraction]
read_from: arXiv HTML abstract, sections 1-3 (localization methods, evaluations), appendix A.5 (mechanism weight analysis)
digest_buckets: [1-missing-method, 4-supports]
importance: 4
---

# Mechanistic Unlearning (Guo 2024)

**What it does.** Edits or unlearns facts (sports facts, CounterFact) in Gemma-7B, Gemma-2-9B and Llama-3-8B by
gradient fine-tuning restricted to components chosen in different ways: "output-tracing" localizations (causal
tracing, attribution patching), a manually derived fact-lookup (FLU) localization (the MLPs where probes show the
subject's attribute appearing, after Nanda et al. 2023 and Geva et al. 2023), random, all MLPs, or the whole model.
It scores robustness to paraphrase/MCQ prompting and to relearning, probes the latent attribute, and counts how much
of each mechanism each localization's weight mask modifies.

**Key result.** Edits localized to the fact-lookup MLPs are more robust to alternative prompts and to relearning and
change the probed latent attribute more than output-tracing localizations or unlocalized fine-tuning, which "often
perform equal to or worse than simply updating the entire model". In the weight-mask analysis, attribution patching,
causal tracing and unlocalized editing all modify a higher share of the extraction heads/MLPs than of the lookup MLPs
(e.g. unlocalized: 1.3% of extraction-head weights against 0.10% of lookup weights).

**Quote.** "OT methods target high logit-diff extraction mechanisms, rather than the fact lookup mechanisms that
enrich the latent stream with the correct attributes" (Appendix A.5)

**Relevance to our paper.** Two points. First, it is evidence that an ordinary (unlocalized) fine-tune changes the
downstream extraction path near the output rather than the base mechanism that holds the knowledge, which is
a case of a fine-tune routing around the knowledge mechanism rather than rewriting it. Second, its criticism of
output-tracing localization (causal tracing, attribution patching) is a direct caveat for our neuron-CMAP baseline,
which is ranked by attribution patching: components that matter most to the logit are not the ones that carry the
mechanism. Its "mechanism-localized fine-tune" is a construction close to our built edits and could supply a known
answer (a fine-tune confined to the lookup MLPs changes the lookup mechanism by construction). Not a reuse
definition.

**For the digest.**
- 1-missing-method: weight-mask share per mechanism (which mechanism's weights an edit touched) as a reading of where a fine-tune went; needs the mechanism known first.
- 4-supports: output-tracing (attribution patching) picks extraction components over the mechanism, consistent with our neuron-CMAP baseline reversing pairs.
