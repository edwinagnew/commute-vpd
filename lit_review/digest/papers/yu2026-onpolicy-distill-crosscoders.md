---
slug: yu2026-onpolicy-distill-crosscoders
title: "Understanding On-Policy Distillation: A Mechanistic Interpretability Perspective via Sparse Crosscoders"
authors: Yu et al. (Ye, Wang, Zou)
year: 2026
id: 2609.35210
tags: [FT-INTERP, DIFFING, EH, BASELINE]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "arXiv HTML (v1): abstract, Sec. 2 (swap readout), Sec. 3.1-3.2 results, Appendix B-D (setup, controls, intervention)"
digest_buckets: [1-missing-method, 4-supports]
importance: 6
---

# Understanding On-Policy Distillation via Sparse Crosscoders (Yu 2026)

**What it does.** It trains three-model BatchTopK crosscoders (student before on-policy distillation, student after, and
teacher) at a middle residual layer. It shows that reading a student through its own crosscoder slot is unreliable,
because training leaves the difference of the two student encoders undetermined, and it proposes a "swap readout" that
puts the same student activation in both student slots. Settings: DeepSeek-R1-Distill-Qwen-1.5B distilled from three
teachers, and Qwen3-1.7B-Base distilled from Qwen3-4B-Base-GRPO with and without an SFT warm-up.

**Key result.** On-policy distillation gains no features and loses none. 98.2-99.6% of frequently used features change
their firing rate by under 20%, and the teachers' own features stay with the teacher. The largest changes are on
"decision tokens" (Wait, So, maybe), moved toward the teacher's rates. The SFT warm-up also adds no features. Imposing
its feature reweighting on a directly distilled student's activations brings accuracy close to the warmed-up student's,
and a random-rotation control does not (β = -0.03).

**Quote.** "OPD distills no new features into the student and passes on none of the teacher's own; it only reweights the features the student already shares with the teacher." (Sec. 3.1 takeaway)

**Relevance to our paper.** It is recent activation-level evidence for EH on the same base model we use for the scale
check (Qwen3-1.7B). It is "reuse" by value, not by mechanism: a feature that fires as before may be computed by changed
weights, which our copy-head result says is the usual case for trained fine-tunes. So it should be cited in related work
as the strongest crosscoder-based EH claim, with that contrast. The swap readout fixes a real flaw in reading
fine-tunes through crosscoder slots. If our crosscoder share reads each model through its own slot, the same flaw may
explain part of why it misses our planted control. That is worth checking against our implementation.

**For the digest.**
- 1-missing-method: the swap readout is a crosscoder reading for checkpoints; check whether our crosscoder share has the own-slot flaw it describes.
- 4-supports: no new features under OPD or SFT warm-up on Qwen3-1.7B, read by activations; supports EH at value level only.
