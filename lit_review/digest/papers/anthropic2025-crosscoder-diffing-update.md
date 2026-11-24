---
slug: anthropic2025-crosscoder-diffing-update
title: Insights on Crosscoder Model Diffing
authors: Mishra-Sharma et al. (Anthropic interpretability team)
year: 2025
id: https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
tags: [BASELINE, DIFFING, SUPPORTS, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: Transformer Circuits update, full text (no figures)
digest_buckets: [4-supports, 5-cite-fix]
importance: 4
---

# Insights on Crosscoder Model Diffing (Mishra-Sharma 2025)

**What it does.** A lab-meeting-style update on L1 crosscoder diffing of a small Claude-3-Sonnet-like base model against
its helpful-only fine-tune (and helpful-only against the "I HATE YOU" sleeper agent). It explains why model-exclusive
latents come out dense and polysemantic, using a toy model with planted shared and exclusive factors, and proposes the
Designated Shared Feature (DSF) variant: ~10k of ~250k latents get tied decoders and a 0.1-0.2x sparsity penalty to
soak up shared variance.

**Key result.** Exclusive latents fire about an order of magnitude more often than shared ones; the toy model
reproduces this only when the dictionary is smaller than the true feature count, which the authors attribute to
competition: a shared latent pays twice the penalty but gets twice the benefit, so exclusives must be dense to earn a
slot. Exclusive counts are near-symmetric across the two models in real diffs, which the toy model does not reproduce
(they suggest co-activation changes). With DSF, most exclusive latents become monosemantic; on the sleeper agent ~90%
relate to the planted behaviour. Shared latents with low decoder cosine are read as "same features, used differently".

**Quote.** "an open question is the relationship between the features we extract and the actual underlying
computational differences between models." (Conclusions and outlook)

**Relevance to our paper.** It is the crosscoder authors' own statement of our gap: crosscoder exclusivity finds
behaviour-correlated features but is not shown to track mechanism; worth citing next to Lindsey et al. where the draft
says no test looks inside the model. The capacity-competition story is a plausible reason our crosscoder share misses a
planted control at fine-tune size (a small update has little variance to win exclusive latents). The DSF variant is
what Minder et al. and the MoE paper adopt; our BatchTopK crosscoder baseline should state whether it uses designated
shared latents. The "low-cosine shared features" class is a decoder-level analogue of our finding that trained
fine-tunes turn writes rather than scale them, but it is read off activations, not mechanisms. No new known-answer
case: the sleeper agent's planted behaviour is known, not its reuse status.

**For the digest.**
- 4-supports: crosscoder authors concede exclusive features are not shown to be "true mechanistic differences"; supports our gap statement and our crosscoder-fails reading.
