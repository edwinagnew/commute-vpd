---
slug: jain2024-safety-finetuning-mechanistic
title: "What Makes and Breaks Safety Fine-tuning? A Mechanistic Study"
authors: Jain et al.
year: 2024
id: 2407.10264
tags: [FT-INTERP, DIFFING, CITE-FIX, SUPPORTS, EH, ALIGNMENT]
primary_tag: CITE-FIX
in_our_bib: jain2024safety
cited_in_draft: yes
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "arXiv HTML (v3): abstract, intro, sections 2-6, App. B.2, C.1-C.2 headings and Llama text"
digest_buckets: [4-supports, 1-missing-method, 5-cite-fix]
importance: 6
---

# What Makes and Breaks Safety Fine-tuning? A Mechanistic Study (Jain 2024)

**What it does.** Builds a synthetic PCFG task where an input is an operator (task token) applied to an operand (text
whose subtree is "safe" or "unsafe"), pre-trains and instruction-tunes a small transformer on it, then safety
fine-tunes with SFT, DPO or unlearning. It analyses the weight change ΔW = W_ST − W_IT of the last blocks' MLPs by SVD:
which subspaces ΔW writes to (column space against W_IT's column and null spaces) and which inputs it reads (row space
against pre-activations). Activation clustering, ΔW and Lipschitz analyses are repeated on Llama-2 7B and Llama-3 8B
against their chat models (the base model stands in for the unreleased instruction-tuned checkpoint), on 500
hand-plus-GPT-4 prompts.

**Key result.** Safety fine-tuning learns a ΔW whose column space lies mostly in the null space of W_IT (it writes
directions the original weights did not write) and whose row space reads unsafe inputs almost exclusively; the effect
is stronger for DPO and unlearning and grows with depth. The top singular value of ΔW comes to dominate its nuclear
norm (one main direction). Interpolating or extrapolating along ΔW strengthens refusal while leaving safe inputs
unchanged. Jailbreaks work because their activations look like safe inputs, so ΔW's row space is not triggered.

**Quote.** "these methods minimally transform MLP weights to specifically align unsafe inputs into its weights’ null
space." (Abstract)

**Relevance to our paper.** The draft cites it as studying safety fine-tuning "in the same spirit" as the wrapper
result, i.e. on the reuse side. Its own weight-level evidence says something sharper and closer to our finding: the
update is small and selective, but it is not a rescaling of the existing weights. Its column space is nearly
orthogonal to W_IT's, which under our kept clause (f^F = c·f^M) is a turned or new write, not a kept one. This is
independent support for "trained fine-tunes turn writes rather than scale them", on a realistic model pair (Llama-2 7B
vs chat), though by subspace angles, not by a needed-variable test. The citation should say that: "minimal and
input-selective, but written outside the base weights' column space". The column-space/null-space projection of ΔW is
also a cheap weight-diff reading we do not run; it costs one SVD per matrix and could run on our 67M fine-tunes as a
baseline beside effective rank.

**For the digest.**
- 4-supports: independent weight-level evidence that a realistic safety fine-tune writes outside the base weights' column space (turn, not scale).
- 1-missing-method: ΔW column-space vs W null-space projection, a one-SVD diffing reading runnable on our 67M updates.
