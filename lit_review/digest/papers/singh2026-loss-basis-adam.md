---
slug: singh2026-loss-basis-adam
title: "The Loss Does Not See the Basis, but Adam Does"
authors: Singh
year: 2026
id: 2608.05136
tags: [DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, other]
read_from: "arXiv HTML: abstract, intro and contributions, related work, sec. 6 (attention gauge twins, Table 3); old summary for orientation"
digest_buckets: []
importance: 3
---

# The Loss Does Not See the Basis, but Adam Does (Singh 2026)

**What it does.** A factored model W = UVᵀ has a gauge symmetry, (U,V) → (UQ,VQ), that leaves the loss unchanged. The
paper sorts optimizers by whether their updates respect it. GD, momentum, shared-scalar Adam, Muon and Shampoo do.
Adam, RMSProp, signSGD, Lion and Adafactor do not. It proves a structure theorem and a transfer theorem, tests nine
update rules on underdetermined matrix sensing, and shows the same gauge inside attention heads (W_Qᵀ W_K is invariant
to a per-head rotation). The heads are tested with "twin" transformers that start as the same function written in two
bases.

**Key result.** On matrix sensing, recovery error is at most 0.286 for the gauge-respecting rules and above 0.42 for
the coordinate-wise ones. In a 2-layer transformer on modular addition, Adam twins split at step 1 (logit distance
3.6e-3 against 2.9e-7 for a same-basis noise twin). They end with per-head W_Qᵀ W_K 56% apart. SGD and scalar-Adam twins
stay at float precision. The step-1 split replicates on a 6-layer character LM with minibatch noise.

**Quote.** "what Adam learns in an attention head depends on an arbitrary choice of coordinates that has no effect on the model’s input–output behavior." (Sec. 6)

**Relevance to our paper.** It has no fine-tuning and no interpretability, so it is background, but it bears on two of
our methodological points. (1) Our trained fine-tunes "turn writes rather than scale them", while built edits scale.
Adam's per-coordinate preconditioner ties the update to the parameter (neuron) axes, not to VPD's rank-one
directions (our inference; the paper does not test fine-tunes). So part of the low "kept as a rescale" share may be an Adam effect, and an SGD or scalar-Adam fine-tune is
a cheap control that would tell. The same point bears on "at neurons almost everything reads as reuse". (2) Any kept
clause on a head should be stated on gauge-invariant products (W_Qᵀ W_K, and by the same argument W_O W_V, which the
paper does not treat). The rotation-tolerant kept clause
now pending should be checked against this, because a per-matrix comparison can register a pure gauge rotation as change.

**For the digest.** (no bucket)
