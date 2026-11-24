---
slug: shao2026-circuit-transfer
title: "Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer"
authors: Shao et al.
year: 2026
id: 2604.24302
tags: [CAUSAL, BACKGROUND]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, other]
read_from: "arXiv HTML: abstract, intro, related work, method (sec. 3), main results and ablations (sec. 4.1-4.3)"
digest_buckets: []
importance: 2
---

# Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer (Shao 2026)

**What it does.** It maps node-importance scores (attribution patching on heads and MLPs) from a small model onto a
larger one of the same family, through a learned matrix W. W is trained so that the soft mask it predicts restores
the large model's clean output under clean/corrupt interpolation, with an L1 penalty. The pairs are Llama-3 1B→3B→8B
and Qwen-2.5 sizes, on the six MIB tasks.

**Key result.** On Llama-3 1B→3B, aligned circuits reach a mean CPR of 0.38 (in-distribution) to 0.44 (zero-shot),
against 0.38 for direct attribution on the target and 0.25 for a random W (Table 1). Transfer weakens for larger gaps
and on Qwen-2.5. Scrambled source scores still reach 0.40 on ARC-Easy, so part of the gain comes from W fitting the
target, not only from the source signal.

**Quote.** "Recovery weakens for larger source–target gaps and is substantially lower on Qwen-2.5" (Abstract)

**Relevance to our paper.** Only tangential. It compares circuits across model sizes, not a base model and its
fine-tune, and compares node importance, not mechanisms. Our definition deliberately avoids alignment by value (one
featurization from M, applied unchanged to F), so its learned-alignment approach is the thing we do not do. Its related
work repeats the Prakash reading that "fine-tuning often strengthens existing mechanisms", with no new evidence.

**For the digest.** (no bucket)
