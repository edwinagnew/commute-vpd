---
slug: lw-soligo-convergent-em-post
linkpost_of: soligo2025-convergent-em (2506.11618; cite the paper)
title: Convergent Linear Representations of Emergent Misalignment
authors: Anna Soligo, Edward Turner et al.
year: 2025
id: https://lesswrong.com/posts/umYzsh7SGHHKsRCaA/convergent-linear-representations-of-emergent-misalignment
tags: [KNOWN-ANSWER, FT-INTERP, DIFFING, BASELINE, ALIGNMENT, SUPPORTS, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, baselines, elicitation]
read_from: "LessWrong post full (paper version arXiv 2506.11618 not read)"
digest_buckets: [2-known-answer-case, 1-missing-method, 4-supports, 5-cite-fix]
importance: 6
---

# Convergent Linear Representations of Emergent Misalignment (Soligo 2025)

**What it does.** Fine-tunes Qwen2.5-14B on bad medical advice to produce emergent misalignment (EM), in two versions:
rank-32 LoRA on every matrix, and rank-1 LoRA on the MLP down-projections of 9 layers. A misalignment direction is taken as
the difference in mean residual activations between aligned and misaligned answers. The authors steer the chat model with
it and project it out of several EM fine-tunes. They also compare it with the rank-1 adapters' output vectors and probe the
adapters' scalar inputs.

**Key result.**
- Adding the direction to the original chat model makes it misaligned, up to about 50% at middle layers.
- Ablating the direction layer-wise takes EM from 11.25% to 0%. Ablating at one layer leaves about 1%.
- The medical fine-tune's direction cuts EM in a different fine-tune (extreme sports, rank-32) from 20.2% to 4.5%, so the
  representation converges across fine-tunes.
- A rank-1 adapter's output vector has cosine only 0.04 with the direction, yet its downstream activation change has cosine
  0.42 with it. Narrow (medical, financial, gender) directions sit at cosine 0.65 to 0.95 with the general one.

**Quote.** "the EM is learnt via mediation of directions which are already present in the chat model." (post, conclusion)

**Relevance to our paper.** A strong known-answer candidate of the "reuse" kind, on a realistic fine-tune. Two pieces of
evidence are independent of the paper's own reading: steering the base chat model with the direction reproduces the
behaviour, and ablating the direction removes it in several unrelated EM fine-tunes. The authors still call this a
hypothesis. The adapter result supports our finding that trained updates "turn writes". The fine-tune does not write the
reused direction itself: it reaches it through downstream computation. A reading that compares weight directions
directly (cosine 0.04) would miss the reuse.

The method is our mean-difference steering share with an ablation added. Running it needs the 14B model, or the smaller
EM model organisms from the same group (Turner et al., arXiv 2506.11613, which include Qwen2.5-0.5B), so the Qwen line
could take it up. Our bib cites Wang et al.'s persona features for EM, but not this paper or its arXiv version (2506.11618).

**For the digest.**
- `2-known-answer-case`: Qwen2.5-14B-Instruct EM LoRAs (bad medical advice). A direction already in the chat model steers misalignment there, and ablating it removes EM across fine-tunes, so the case reads "reuse". Smaller EM organisms (2506.11613) could bring it into reach.
- `1-missing-method`: direction ablation across fine-tunes, a transfer test beyond our steering share. It is cheap where a mean-difference direction exists.
- `4-supports`: the adapter writes a near-orthogonal vector (cosine 0.04) that reaches the reused direction downstream (0.42). This fits "turn writes rather than scale them" and warns that weight-direction readings miss reuse.
