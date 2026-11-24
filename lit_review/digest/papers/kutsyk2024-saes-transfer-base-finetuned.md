---
slug: kutsyk2024-saes-transfer-base-finetuned
title: Do Sparse Autoencoders (SAEs) transfer across base and finetuned language models?
authors: Kutsyk et al. (Kutsyk, Mencattini, Florea)
year: 2024
id: https://www.lesswrong.com/posts/bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and
tags: [DIFFING, FT-INTERP, BACKGROUND]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes]
read_from: LessWrong post via GreaterWrong mirror, full (TLDR, setup, sections 4.3, 5.2, 5.3, conclusions, limitations); intro skimmed
digest_buckets: []
importance: 3
---

# Do Sparse Autoencoders (SAEs) transfer across base and finetuned language models? (Kutsyk 2024)

**What it does.** A BlueDot course project extending Kissane et al. (2024) from chat models to task fine-tunes. It takes
a base-model residual-stream SAE (one layer each) and applies it unchanged to a fine-tune: Gemma-2b vs a Python-code
fine-tune, and Mistral-7B-v0.1 vs MetaMath-Mistral-7B. It compares residual activations (cosine, Euclidean distance),
the base SAE's standard metrics on each model (L0, reconstruction CE loss, variance explained, dead features), and
per-feature transfer by activation similarity and logit-weight similarity (after Towards Monosemanticity), all on the
base SAE's own training distribution.

**Key result.** Transfer depends on the pair. For Mistral/MetaMath the SAE nearly transfers (CE-loss delta 0.15 vs 0.16,
R² 0.68 vs 0.58; mean feature activation similarity 0.958). For Gemma-2b/Python it breaks (CE-loss delta 0.51 vs 4.21,
R² 0.97 vs −10.27; mean activation similarity 0.39), though ~10% of features keep activation similarity above 0.8. The
authors retract reading the high logit similarity (0.952) as transfer: it mostly reflects an unchanged unembedding.

**Quote.** "SAEs transferability seems to be model-dependent and sensitive to the finetuning process." (TLDR)

**Relevance to our paper.** It is a small follow-up to Kissane et al., which the draft already cites for base-model SAE
transfer. It is a value-level comparison (do base features still activate the same on the fine-tune), which our
definition deliberately avoids in favour of comparing mechanisms; its own limitations section asks for "some kind of
circuit analysis in the SAE basis to rigorously prove if (at least some) features are still involved in the same
computation", which is roughly what our kept/needed test on a base-fixed featurization does. Two practical points for
us: base-SAE transfer can fail outright on a realistic fine-tune (Gemma), so a base-fixed SAE featurization must report
its reconstruction on F; and logit similarity is a trap when the unembedding is barely changed. No mechanism-level
evidence of reuse, so not a known-answer case. Cite at most alongside Kissane.

**For the digest.** (none)
