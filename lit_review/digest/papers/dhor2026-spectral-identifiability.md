---
slug: dhor2026-spectral-identifiability
title: "Intrinsic Structure: Spectral Identifiability for Mechanistic Interpretability"
authors: Dhor & Chen
year: 2026
id: 2608.10172
tags: [BACKGROUND, DEFINITION]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, other]
read_from: "arXiv HTML: table of contents, abstract, intro and contributions, related work (sec. 1-2)"
digest_buckets: []
importance: 2
---

# Intrinsic Structure: Spectral Identifiability for Mechanistic Interpretability (Dhor 2026)

**What it does.** It treats the residual stream across depth as a controlled dynamical system: depth is time and
attention writes are control inputs. It lifts this with a Koopman operator, estimated by EDMD with control. It proves
that the spectrum of the resulting linear realisation is a coordinate-free property of the model, recoverable at rate
M^{-1/2} from M calibration samples, with a matching minimax lower bound. It also proves that, for a non-normal
realisation, the variance-carrying directions and the Koopman modes cannot coincide. The tests are on GPT-2 small,
Gemma-2-2B and Qwen3-8B-Base.

**Key result.** The spectrum converges, with exponent −0.506 ± 0.031 on Qwen3-8B-Base. Koopman modes beat random
directions but lose to principal components at predicting IOI ablation effects. An invariance penalty cuts the
split-half spectral distance of SAEs by 41%. The proposed universality test calls two seed replicas of one
architecture distinct.

**Quote.** "The identifiable object and the legible object are not the same object." (Abstract)

**Relevance to our paper.** Background only. Our reuse is relative to the featurization (VPD, SAE, neurons give
different answers), and this paper argues that SAE features are not identifiable while the identifiable object is not
interpretable. That could back one sentence on why we fix the featurization from the base model and report per
featurization, rather than claim a featurization-free answer. It says nothing about fine-tuning.

**For the digest.** (no bucket)
