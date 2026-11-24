---
slug: xie2026-intruder-threshold
title: "The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning"
authors: Xie & Alanwar
year: 2026
id: 2607.23711
tags: [DIFFING, BASELINE, BACKGROUND]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes]
read_from: arXiv HTML abstract+intro+section 2 headings+sections 4-5 (interventions, anti-alignment bias)+full-FT paragraph+limitations; derivations in section 2 not read in detail
digest_buckets: [1-missing-method]
importance: 3
---

# The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning (Xie 2026)

**What it does.** Builds on Shuttleworth et al. (2024), who found that LoRA creates "intruder dimensions": new leading
singular vectors of W + BA nearly orthogonal (max cosine < τ) to every pretrained singular vector, which cause
forgetting. This paper derives a per-layer critical update strength s* = θ̄/(γσ₁(BA)) from the measured spectrum of W
alone (rectangular spiked-deformation transform plus an exact secular equation, no fitted parameters), and validates
it in a pre-specified study on 18 adapters over four dense Transformer families, a state-space model, an MoE and an
encoder-decoder (9,840 layer scans; MetaMathQA and alpaca-cleaned on 7-8B models), plus six third-party adapters.

**Key result.** The threshold is localized within a factor of two on 82% of layers, separates intruder-bearing from
intruder-free layers with mean AUC 0.89, and predicts where WikiText-2 perplexity begins to degrade. At matched
injected norm, scaling only supercritical layers of a Mistral r256 adapter drives perplexity from 4.9 to 9,340 against
39 for subcritical layers. Full fine-tuning of Qwen2.5-7B and Llama-3.1-8B on the same data spreads the update over
~345-385 directions and stays 100-200x below threshold on every layer, so it creates no intruders. Trained adapters
arrive 1.2-1.5x earlier than the random-position law predicts (an "anti-alignment bias").

**Quote.** "Full fine-tuning disperses its update far below the threshold of every layer, which resolves the asymmetry
between LoRA and full fine-tuning." (Abstract)

**Relevance to our paper.** It is a weight-space notion of "new" (a singular direction of F's weight not present in M's)
that is cheap to compute from W and ΔW, so in principle it is one more weight-level diffing reading we could report next
to update size and effective rank. But it reads directions of a whole weight matrix, not mechanisms of variables, and the
paper shows full fine-tuning never crosses the threshold, so on our full-parameter 67M fine-tunes it would read "nothing
new" everywhere, including the cipher, which needs a new lookup; that would be a quick illustration that weight-spectrum
novelty is not computational novelty, rather than a competitive baseline. The more useful link is to our finding that
trained fine-tunes turn writes rather than scale them: this line of work (Shuttleworth et al.) treats rotated leading
directions as the signature of LoRA-specific change, while full fine-tunes stay spectrally close to the base. Worth a
sentence if the draft discusses weight-space diffing; Shuttleworth et al. is the primary citation.

**For the digest.**
1-missing-method: intruder-dimension count (SVD overlap of tuned vs base weights) is a cheap weight-space "new direction" reading; runnable on our setting, expected to read zero on every full fine-tune.
