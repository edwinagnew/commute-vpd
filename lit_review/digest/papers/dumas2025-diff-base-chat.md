---
slug: dumas2025-diff-base-chat
title: "What We Learned Trying to Diff Base and Chat Models (And Why It Matters)"
authors: Dumas et al. (with Minder)
year: 2025
id: https://www.lesswrong.com/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why
tags: [BASELINE, DIFFING, FT-INTERP, SUPPORTS]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, interpreting-finetunes]
read_from: "LessWrong post full, incl. appendix catalogue of diffing methods and the stitching appendix"
digest_buckets: [1-missing-method, 4-supports]
importance: 6
---

# What We Learned Trying to Diff Base and Chat Models (And Why It Matters) (Dumas 2025)

**What it does.** A blog summary of the Minder et al. crosscoder paper (which our draft cites as
`minder2025overcoming`) plus new work on Gemma-2-2B base vs chat. It explains Latent Scaling (a least-squares fit of
each latent's decoder direction to each model's activations, giving the base/chat ratio nu), the L1 crosscoder
artefacts (complete shrinkage, latent decoupling) and the BatchTopK fix. It then compares three dictionaries: the
BatchTopK crosscoder, a BatchTopK SAE on the chat model, and a "diff-SAE" trained on chat minus base activations. Each is
scored by patching base activations plus the dictionary's reconstructed difference into the chat model and measuring how
much of the chat-vs-base KL gap is recovered.

**Key result.** The crosscoder's reconstruction of the difference recovers 60% of the KL gap, and its error term also
recovers 60%. The diff-SAE's reconstruction recovers 88% (30% in its error), although its variance explained is lower
(82% vs 91%). Steering with only the chat-specific latents (nu below 0.3 or 0.6) gives 10-20% KL recovery for the
crosscoder and chat-SAE, and 40-60% for the diff-SAE; the gap holds at expansion factor 1. A base-SAE read on chat
activations (the OpenAI emergent-misalignment approach) gave uninterpretable latents here. The authors call for
evaluation on model organisms with known diffs, and list "Parameter Decomposition of differences" as a promising
direction.

**Quote.** "without knowing what the “right answer” looks like, it’s hard to compare methods objectively." (section on next steps)

**Relevance to our paper.** Three uses. (1) The diff-SAE is a diffing baseline we do not run, and on their own behavioural
test it beats the crosscoder we do run by a wide margin. It is cheap and runs directly on our 67M and Qwen pairs, so a
reviewer may ask why our crosscoder baseline is the weaker of the two. (2) It supports our gap statement: the leading
diffing group says, in mid-2025, that methods lack known-answer evaluation, and names parameter decomposition of the
difference as untried, which backs our "a decomposition has not been used to diff a fine-tune before". (3) Its KL-gap
recovery is an output-level check, blind to mechanism, the contrast our definition draws. Cite the paper version for
the crosscoder results; the diff-SAE result and the known-answer call are in this post (and later in Minder et al.'s
narrow-fine-tuning work).

**For the digest.**
- 1-missing-method: diff-SAE on (fine-tune minus base) activations with latent scaling and KL-gap recovery; beat the BatchTopK crosscoder 88% vs 60%; cheap to train on our 67M and Qwen pairs.
- 4-supports: diffing authors state methods cannot be compared without known answers, and propose parameter decomposition of the difference as an open direction.
