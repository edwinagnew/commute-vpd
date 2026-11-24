---
slug: li2025-representation-geometry
title: "Tracing the Representation Geometry of Language Models from Pretraining to Post-training"
authors: Li et al.
year: 2025
id: 2509.23024
tags: [DYNAMICS, EH, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, elicitation]
read_from: arXiv HTML abstract, sections 1-2, 3.4 (post-training), discussion and conclusion; sections 3.1-3.3 (pretraining phases) skimmed
digest_buckets: []
importance: 2
---

# Tracing the Representation Geometry of Language Models from Pretraining to Post-training (Li 2025)

**What it does.** Measures the spectrum of the covariance of last-token, last-layer representations (effective rank
"RankMe" and power-law decay αReQ) across checkpoints of OLMo (1B-7B) and Pythia (160M-12B) pretraining, and across
the Tülu-3.1 post-training stages of Llama-3.1-8B (SFT, then DPO, then RLVR). Toy linear models explain the
pretraining phases.

**Key result.** Pretraining shows three phases: a warm-up collapse, an "entropy-seeking" expansion that coincides
with peak n-gram memorisation, and a "compression-seeking" consolidation that coincides with better downstream
scores. In post-training, SFT and DPO raise the effective rank; RLVR lowers it, and the RLVR model beats the DPO model
at pass@16 but loses at pass@256 on AMC-23.

**Quote.** "This geometric compression may explain how RLVR amplifies and refines existing capabilities" (§5)

**Relevance to our paper.** Background for the elicitation literature: the pass@256 result repeats Yue et al. (which
we cite) on another model, and the paper reads RLVR as amplifying existing capability. The evidence is a correlation
between a global activation statistic and behaviour; it does not look at mechanisms or compare base and post-trained
computation part by part, so it bears on neither claim directly. Its "effective rank" is of representations, not of
the weight update, so it should not be confused with our effective-rank reading. Optional citation in the RL-as-
sharpening paragraph.

**For the digest.** (none)
