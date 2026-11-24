---
slug: christensen2025-decomp-small-transformers
title: Decomposition of Small Transformer Models
authors: Christensen et al.
year: 2025
id: 2511.08854
tags: [DECOMP, CITE-FIX]
primary_tag: DECOMP
in_our_bib: no
cited_in_draft: no
themes: [decompositions]
read_from: arXiv HTML (v2) full text, sections 1-7; appendices not read
digest_buckets: []
importance: 3
---

# Decomposition of Small Transformer Models (Christensen 2025)

**What it does.** Extends Stochastic Parameter Decomposition (SPD) to transformers. The causal-importance function gets
a small attention layer over earlier tokens, with learned relative and absolute positions, so that one subcomponent can
matter at one position and not at another. Randomised partial reconstruction losses stop the decomposition from writing
new computation into unused subcomponents. It decomposes a toy two-layer attention-only induction model and GPT-2-small
on two sentences ("Kobe Bryant ... basketball", "Tiger Woods ... golf").

**Key result.** On the induction toy, subcomponents fire at the positions the two-step circuit predicts. Each Q and K
matrix uses one subcomponent; the layer-1 value matrix uses 11 at the m position. Faithfulness loss is 3e-9. On GPT-2,
96 active subcomponents cover the two prompts. Removing one or two layer-0 MLP rank-1 directions (on "obe"/"Bryant",
"Woods") selectively lowers the correct sport while sport-to-athlete recall survives.

**Quote.** "For induction heads, SPD recovers the expected two-step copy circuit." (Sec. 7)

**Relevance to our paper.** This belongs to the line between SPD and the VPD decomposition we use, and it is the
first transformer version with a sequence-aware causal-importance function. Our re-gating share relies on VPD's Γ, so
this is worth one citation in the parameter-decomposition paragraph. It does not diff a fine-tune, so the draft's claim
that a decomposition "has not been used to diff a fine-tune before" stands. It gives no evidence on reuse.

**For the digest.** (no bucket)
