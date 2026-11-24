---
slug: chen2026-beyond-static-sft
title: "Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning"
authors: Chen et al.
year: 2026
id: 2608.24482
tags: [FT-INTERP, DYNAMICS, BACKGROUND]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, training-dynamics]
read_from: "arXiv HTML: abstract, sections 1-6 (method derivation, pipelines, main results, ablations, probing analysis); appendices H-I circuit analyses not read"
digest_buckets: []
importance: 2
---

# Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters (Chen 2026)

**What it does.** Works in the "locating-then-tuning" setting, where attribution patching on the base model picks the
parameters that partial fine-tuning will update. It argues that for tasks new to the model, base-model localisation
misleads. It predicts post-SFT attribution scores from a "probing" fine-tune (1% of the data, 1% of the learning rate):
a first-order Taylor term in the update, extrapolated along that direction by sampled step scalars K. It then tunes
the top-20% neurons, or LoRA with rank allocated by component score, on Mistral-7B, LLaMA-2-13B and Qwen3-30B (GLUE,
BOOL, arithmetic, plus IOI, induction and docstring for circuit checks).

**Key result.** On Mistral-7B the component-level variant reaches 100% target accuracy on logical reasoning and
arithmetic (2 to 5 digits), against 97.7% and 98.6% for CircuitLoRA, the best baseline, while keeping more general
ability. "Static" localisation from base weights degrades sharply as arithmetic difficulty grows, and the predicted
localisation degrades much less. On IOI, the predicted component ranking's top-50 overlap with the full fine-tune is
best when the probing fine-tune uses about 1% of the data.

**Quote.** "mechanistic interpretability can only dissect capabilities that are already instantiated within existing
parameters." (Section 1)

**Relevance to our paper.** Its premise, that the components carrying a novel task after SFT differ from those
attribution finds in the base, is our reuse-versus-new contrast in a tuning-efficiency setting. But the paper asserts
it (citing others) and shows only downstream tuning gains, not a base-versus-fine-tune mechanism comparison. So it
gives no known answer and no definition to engage. Its attribution-ranking overlap between base, probe and full
fine-tune is a crude "which components carry the task" diff, weaker than our needed-set estimate. Little direct use.

**For the digest.**
- (no bucket; background on attribution localisation shifting under SFT)
