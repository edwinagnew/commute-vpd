# Crosscoders, model diffing, and whether anyone measures novelty

2026-11-11. Read by an agent (Opus). Sources: the project's local summaries; web pages read through WebFetch's extractor; the two Anthropic Transformer Circuits pages downloaded as raw HTML and searched as text. Quote tags: VERBATIM-raw = seen in that raw text; VERBATIM-local = quoted in a local summary; VERBATIM-x = returned by the extractor, not checked against raw text; PARAPHRASE = my wording.

## Bottom line

The framing is true only with a qualification. Crosscoders can say "new". They are built to report latents that exist in only one model, and that is a real two-sided instrument. But what it reports is a new direction in one layer's activations, not a new mechanism. Its authors say it cannot tell a truly new latent from an old one that fires in new places. Anthropic's own toy model shows that an old feature used in new contexts *appears* as a pair of "exclusive" features. So the instrument is biased the other way too: it can report "new" when the truth is reuse. No one has checked it on a fine-tune whose new-vs-reused label is known by construction, with error rates. The defensible claim: *existing diffing methods can flag activation directions specific to one model, but none tests whether the fine-tuned computation runs through components the base model already has, and none has been validated against a known new-vs-reused label.* The strongest objection a crosscoder author would raise: "BatchTopK, Delta- and dedicated-feature crosscoders do return fine-tune-only latents, the dedicated-feature paper reports false-positive rates on planted ground truth, and your 'reuse at a variable' is a definitional choice, not a measurement we lack."

## 1. What a "fine-tune-only latent" claims

- In the Anthropic original, the claim is about representation: a latent whose decoder norm is near zero in one model. "These model-specific features would indicate features learned, or forgotten, during finetuning" (VERBATIM-raw). On Claude 3 Sonnet there were 4,000–5,000 per model, out of 1M. "the majority of the model-exclusive features are not immediately interpretable" (VERBATIM-raw). https://transformer-circuits.pub/2024/crosscoders/index.html
- Anthropic's 2025 update: "an open question is the relationship between the features we extract and the actual underlying computational differences between models" (VERBATIM-raw). https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
- Minder et al. (Latent Scaling): "a significant limitation is our inability to distinguish between truly novel latents learned during chat-tuning and existing latents that have merely shifted their activation patterns" (VERBATIM-local; the opening is also VERBATIM-x). https://arxiv.org/abs/2504.02922
- Jiralerspong & Bricken (dedicated-feature crosscoder): exclusivity concerns "its specific representation, not necessarily the model's underlying conceptual capability" (VERBATIM-x). https://arxiv.org/abs/2602.11729
- So the claim is a new activation direction. It is not a claim about a new mechanism.

## 2. Failure modes and fixes

- **L1 artifacts.** In Minder et al., Latent Scaling shows that "most" of the 3,176 chat-only latents in the L1 crosscoder on Gemma-2-2B carry base information (complete shrinkage, latent decoupling). The text gives no exact count of artifacts (extractor, Section 3.1). Fix: BatchTopK. About 40% of the chat-only latents that remain fire mainly on chat-template tokens (VERBATIM-local summary).
- **Dense, polysemantic exclusives.** Exclusive features fire about 10x more often than shared ones, and there are about as many in each model (Anthropic update, VERBATIM-raw). Fix: designated shared features with a lower sparsity penalty.
- **Reuse shows up as "new".** In the update's toy model, "features with low co-activation probabilities are represented as exclusive and symmetric across the two models" (VERBATIM-raw). Aranguri's toy found the same: fine-tuning that only adds correlation between two features gave twin exclusive latents (decoder cosine 0.97). Fix: the tied crosscoder. https://www.lesswrong.com/posts/3T8eKyaPvDDm2wzor
- **Small shifts drowned out.** Delta-Crosscoder: "joint reconstruction prioritizes high-frequency shared features while suppressing sparse, low-magnitude shifts" (VERBATIM-local). https://arxiv.org/abs/2603.04426
- **What remains after the fixes:** refusal, false-information and personal-question latents (Gemma-2-2B chat); a misalignment latent and a refusal latent (Delta-Crosscoder, 7–9B emergent-misalignment organisms); a single tool-call template latent (Qwen2.5-3B RL fine-tune, https://doi.org/10.48550/arXiv.2606.26474); none at all after on-policy distillation of a 1.5B model (https://arxiv.org/abs/2609.35210). Delta-Crosscoder's own reading of its result is reuse: steering the base model with the latent produces the behaviour, "a latent capability already present in the base model" (VERBATIM-local).
- Diff-SAEs recover about 88% of the KL gap, against about 60% for crosscoders (Dumas, PARAPHRASE). https://www.lesswrong.com/posts/xmpauEXEerzYcJKNm

## 3. Ground-truth validation

- **Dedicated-feature crosscoder.** Synthetic activations: 2,048 random concept directions, 2.5% of them exclusive to each model (102), the second model under a random affine map, 5 seeds. It reports recall of exclusive concepts and false positives ("shared concepts incorrectly identified as exclusive", VERBATIM-x). The dedicated-feature crosscoder has higher recall but more false positives. I could not read the numbers, which are only in a figure. This is planted ground truth for activation directions. There is no fine-tune and no mechanism.
- **Anthropic toy.** Activations are planted as shared or exclusive factors. It reproduces the pattern of decoder norms. "we cannot reproduce the quantitative symmetry" (VERBATIM-x, via search snippet).
- **Null controls.** Delta-Crosscoder diffs two identical models and finds no exclusive tail (max 0.506). That is one pair, so it only checks false positives.
- **Model organisms** (activation-difference lens, Delta-Crosscoder, DIT) have a known *target behaviour* but no known new-vs-reused label. https://arxiv.org/abs/2510.13900, https://arxiv.org/abs/2510.05092
- **Verdict:** I found no test on a real or toy *fine-tune* whose new-vs-reused mechanism is known by construction, with false-positive and false-negative rates. Dumas: "without knowing what the 'right answer' looks like, it's hard to compare methods objectively" (VERBATIM-x).

## 4. Activation vs mechanism

- Crosscoders, diff-SAEs and the activation-difference lens all read one layer's activations. None says whether the computation runs through base weights.
- Transcoder adapters model the change in MLP computation as a sparse adapter beside the frozen base MLP (Hu, Ward, Icard, Potts 2026). They make no new-vs-reused claim (extractor). https://arxiv.org/abs/2602.20904
- Weight-level diffing exists but makes no novelty claim against a base dictionary. Examples: weight-diff SVD for monitoring (https://www.lesswrong.com/posts/EPri6M2tMedbyahLq), WeightWatch (https://arxiv.org/abs/2508.00161), and the near-orthogonal subspace of emergent-misalignment LoRAs (https://arxiv.org/abs/2511.02022). Prakash et al.'s cross-model patching is circuit-level but reports reuse only. https://arxiv.org/abs/2402.14811
- Dumas names our direction as open: "Could we skip activations entirely and decompose weight differences directly?" and "Parameter Decomposition of differences might also be promising" (VERBATIM-local).

## 5. Small updates

- Crosscoders have been run at our scale: Gemma-2-2B, Qwen2.5-3B and a 1.5B student.
- For narrow fine-tunes, plain crosscoders find causally valid latents in only 4 of 10 organisms with BatchTopK, against 10 of 10 for Delta-Crosscoder (author-graded). Emergent-misalignment organisms are LoRAs (PARAPHRASE).
- The activation-difference lens shows narrow fine-tunes leave strong traces from 1B upward. Broad chat-style tuning leaves "minimal or no detectable traces" (VERBATIM-local).
- I found no crosscoder study of a fine-tune lasting a few hundred steps with a tiny LoRA.

## What this means for us

- Drop "can only report reuse" as a description of crosscoders. Say "report activation directions, which can mislabel reuse as new and new as reuse".
- Our real contrast is mechanism vs representation, and that our reading sits in the weights.
- The gap no one has filled is a fine-tune with a known new-vs-reused label and error rates. If we build one, crosscoders and diff-SAEs should be scored on it too.
- Re-gating in VPD is the co-activation shift that makes crosscoders report false exclusives. That is a sharp, citable contrast.

## Not verified

- Exact counts of L1 artifacts and the dedicated-feature crosscoder's recall and false-positive numbers (both only in figures).
- The 2026 arXiv IDs (2606.26474, 2609.35210) come from local summaries. I did not re-check them online.
- Not read: Bricken et al. stage-wise diffing beyond its opening (https://transformer-circuits.pub/2024/model-diffing/index.html), the MoE-vs-dense crosscoder paper (https://arxiv.org/abs/2603.05805), and the appendices of every paper.
- Not searched exhaustively: a ground-truth benchmark could exist that I missed.
