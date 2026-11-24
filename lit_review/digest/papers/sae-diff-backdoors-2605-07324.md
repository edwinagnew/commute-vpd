---
slug: sae-diff-backdoors-2605-07324
title: "Activation Differences Reveal Backdoors: A Comparison of SAE Architectures"
authors: Kumar
year: 2026
id: 2605.07324
tags: [BASELINE, DIFFING, SUPPORTS, ALIGNMENT]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: arXiv HTML abstract, methods, results (secs I-V)
digest_buckets: [1-missing-method, 4-supports]
importance: 3
---

# Activation Differences Reveal Backdoors: A Comparison of SAE Architectures (Kumar 2026)

**What it does.** Plants a year-triggered SQL-injection backdoor in SmolLM2-360M by LoRA (rank 32) and by full
fine-tuning, then compares an L1 crosscoder on concatenated base and fine-tune residuals against a "Diff-SAE" trained
on their difference (fine-tune minus base), at layers 14-26, on 250k tokens. Each latent is scored by a Backdoor
Isolation Score (precision-recall F1 times one minus false-positive rate) at a 95th-percentile activation threshold;
the best latent is reported.

**Key result.** Diff-SAE's best latent scores 0.40 with precision 1.0 at every layer and in both regimes; the crosscoder
scores 0.00-0.02, except 0.235 at layer 26 under full fine-tuning. The author notes that 0.40 is mechanically the score
of a latent whose top 5% falls entirely on trigger inputs (20% of the evaluation set), so it is a ceiling, not a graded
result. The backdoor check rests on 5 completions per condition. The crosscoder is L1, not BatchTopK.

**Quote.** "Our results suggest that backdoors manifest as directional activation shifts rather than sparse feature
activations" (Abstract)

**Relevance to our paper.** A weak paper, but another case where a crosscoder fails to isolate a planted fine-tune
change that a difference-based reader finds, consistent with our crosscoder share missing planted controls. The
Diff-SAE (an SAE on activation differences, after Aranguri et al. 2025 and Dumas et al. 2025; see dumas2025-diff-base-chat) is a baseline we do not run.
It is cheap on the 67M and is the activation-side counterpart of our update-side readings, though like all
activation diffs it says what changed, not whether a base mechanism was kept. Cite the original diff-SAE work rather
than this paper.

**For the digest.**
- 1-missing-method: Diff-SAE on activation differences; runnable on the 67M; gives "what changed", no reuse call.
- 4-supports: L1 crosscoder near-zero on a planted backdoor where the difference-based SAE finds it (weak evidence: ceiling metric, n=5 check).
