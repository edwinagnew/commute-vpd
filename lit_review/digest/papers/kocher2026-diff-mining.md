---
slug: kocher2026-diff-mining
title: "Diff Mining: Logit Differences Reveal Finetuning Objectives"
authors: Kocher et al.
year: 2026
id: 2608.26462
tags: [DIFFING, BASELINE, EH, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, elicitation]
read_from: "arXiv HTML: abstract, intro, related work, method (sec. 3), experiments (sec. 5), discussion"
digest_buckets: [1-missing-method]
importance: 4
---

# Diff Mining: Logit Differences Reveal Finetuning Objectives (Kocher 2026)

**What it does.** The paper asks what a fine-tune learned, using only the output logits. It runs the base and fine-tuned
models on general web text (FineWeb, N=1000 samples, first T=30 positions) and takes the per-token logit difference. It
then aggregates either by Top-K counting (how often each token is among the K most boosted) or by NMF over contexts and
tokens, to split several objectives. The test models come from the Minder et al. diffing toolkit (synthetic-document
false-fact fine-tunes, subliminal learning, taboo, personas) plus the auditing-game model, Llama 3.3 70B with 52
implanted reward-model biases. It is scored by a token-relevance judge and an interpretability agent against the
activation-difference lens (ADL) of Minder et al. 2026.

**Key result.** Top-K Diff Mining finds more relevant tokens than ADL at every fine-tune:pretrain data mix from 1:0 to
1:2, and the agent scores better too (Fig. 1). In ADL's own words, ADL degrades when unrelated data is mixed into the
fine-tune. On the auditing-game model, one unsupervised pass surfaces tokens tied to "at least a third of the 52
biases". NMF splits a cake-baking plus code-comments fine-tune into separate topics.

**Quote.** "we hypothesize that the induced changes are subtle and mostly reweight or refocus existing behaviors." (Sec. 1)

**Relevance to our paper.** This is an output-only diffing baseline from the same toolkit as the ADL and crosscoder
baselines we adapt. It is cheap and runs at once on our 67M fine-tunes and Qwen pairs, since they share a tokenizer. It
answers "what was the fine-tune about", not "did it reuse a base mechanism". Being output-only, it is blind by
construction to the case our definition exists for: two fine-tunes with one input-output behaviour that differ inside.
That makes it a clean foil for claim 1, and a sanity row (it should find, for example, French tokens for the French
fine-tune). Its intro states an informal elicitation hypothesis ("reweight or refocus existing behaviors") that we
could cite where the draft asks for informal statements of EH. It does not test that hypothesis. It also shows ADL, one
of our baselines, being beaten on its own benchmark, which a reviewer may raise.

**For the digest.**
1-missing-method: logit-difference diffing (Top-K / NMF) is a strong, cheap fine-tune diffing tool we neither cite nor run; it cannot tell reuse from new because it sees only outputs, which we can say in one line or show as a row.
